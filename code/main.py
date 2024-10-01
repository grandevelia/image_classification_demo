import gnureadline
import os
import time
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import argparse 

import transforms
from dataloader import ImagenetteLoader
from utils import AverageMeter, LinearWarmupCosineAnnealingLR, get_val_transforms, get_train_transforms

from models import default_cnn_model

parser = argparse.ArgumentParser(description="PyTorch Image Classification")
parser.add_argument("data_folder", metavar="DIR", help="path to dataset")
parser.add_argument("name", metavar="NAME", help="model name")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)",)
parser.add_argument("--epochs", default=90, type=int, metavar="E", help="number of total epochs to run")
parser.add_argument("--warmup", default=5, type=int, metavar="W", help="number of warmup epochs")
parser.add_argument("--start-epoch", default=0, type=int, metavar="E0", help="manual epoch number (useful on restarts)",)
parser.add_argument("-b", "--batch-size", default=256, type=int, metavar="B", help="mini-batch size (default: 256)",)
parser.add_argument("--lr", "--learning-rate", default=0.1, type=float, metavar="LR", help="initial learning rate", dest="learning_rate",)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay",)
parser.add_argument("-p", "--print-freq", default=10, type=int, help="print frequency (default: 10)")
parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)",)
parser.add_argument("--use-resnet18", action="store_true", help="Use pretrained resnet18 model")
parser.add_argument("-k", "--fold-index", default=0, type=int, help="Current fold", dest="fold_index")
parser.add_argument("--folds", default=-1, type=int, help="Total number of folds")

def main(args):
    best_acc1 = 0.0
    fixed_random_seed = 2022
    torch.manual_seed(fixed_random_seed)
    np.random.seed(fixed_random_seed)
    random.seed(fixed_random_seed)

    # set up the model + loss
    if args.use_resnet18:
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 10)
    else:
        model = default_cnn_model(num_classes=10)
    
    model_arch = "bottleneck"
    criterion = nn.CrossEntropyLoss()

    model = model.cuda()
    criterion = criterion.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            model.load_state_dict(checkpoint["state_dict"])
            model = model.cuda()
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {}, acc1 {})".format(
                    args.resume, checkpoint["epoch"], best_acc1
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # set up transforms for data augmentation
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = get_train_transforms(normalize)
    val_transforms = get_val_transforms(normalize)
    
    print("Training time data augmentations:")
    print(train_transforms)
    
    train_dataset = ImagenetteLoader(args.data_folder, split="train", transforms=train_transforms, folds=args.folds, k=args.fold_index)
    val_dataset = ImagenetteLoader(args.data_folder, split="val" if args.folds > 0 else "test", transforms=val_transforms, folds=args.folds, k=args.fold_index)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
        drop_last=False,
    )

    if args.resume and args.evaluate:
        print("Testing the model ...")
        cudnn.deterministic = True
        validate(val_loader, model, -1, args)
        return

    cudnn.enabled = True
    cudnn.benchmark = True

    scheduler = LinearWarmupCosineAnnealingLR(optimizer, args.warmup * len(train_loader), args.epochs * len(train_loader))

    print("Training the model ...")
    train_accs, losses = [], []
    val_accs = []
    for epoch in range(args.start_epoch, args.epochs):
        curr_train_accs, curr_losses = train(train_loader, model, criterion, optimizer, scheduler, epoch, args)
        train_accs += [curr_train_accs]
        losses += curr_losses

        curr_val_accs = validate(val_loader, model, epoch, args)
        acc1 = curr_val_accs.loc[0, 'val_top1']
        val_accs += [curr_val_accs]
        # save checkpoint
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
                "epoch": epoch + 1,
                "model_arch": model_arch,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
            },
            file_folder=f"../models/{args.name}", filename=f"checkpoint.pth.tar"
        )
    train_accs = pd.concat(train_accs)
    losses = pd.concat(losses)
    val_accs = pd.concat(val_accs)
    train_accs.to_csv(f"../models/{args.name}/train_accuracy.tsv", sep="\t", header=True, index=False)
    losses.to_csv(f"../models/{args.name}/losses.tsv", sep="\t", header=True, index=False)
    val_accs.to_csv(f"../models/{args.name}/validation_accuracy.tsv", sep="\t", header=True, index=False)


def save_checkpoint(state, file_folder="../models/", filename="checkpoint.pth.tar"):
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, filename))
    state.pop("optimizer", None)
    torch.save(state, os.path.join(file_folder, "model_best.pth.tar"))


def train(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    num_iters = len(train_loader)

    # set up meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    losses_out = []
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        optimizer.zero_grad()

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.2f} ({loss.avg:.2f})\t"
                "Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t"
                "Acc@5 {top5.val:.2f} ({top5.avg:.2f})".format(
                    epoch + 1,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )
            curr_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.learning_rate
            losses_out.append(pd.DataFrame(data=[{"train_loss": losses.val, "i": epoch * num_iters + i, "lr": curr_lr}]))
            break
    if scheduler is not None:
        print(
            "[Training]: Epoch {:d} finished with lr={:f}".format(
                epoch + 1, scheduler.get_last_lr()[0]
            )
        )
    top_out = pd.DataFrame(data=[{"top1": top1.avg, "top5": top5.avg, "epoch": epoch}])
    return top_out, losses_out


def validate(val_loader, model, epoch, args):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    
    with torch.set_grad_enabled(False):
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=False)
            target = target.cuda(non_blocking=False)
            output = model(input)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t"
                    "Acc@5 {top5.val:.2f} ({top5.avg:.2f})".format(
                        i, len(val_loader), batch_time=batch_time, top1=top1, top5=top5
                    )
                )

    print("******Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5))

    return pd.DataFrame(data=[{"val_top1": top1.avg, "val_top5": top5.avg, "epoch": epoch}])


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
