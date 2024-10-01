import gnureadline
import pandas as pd 
import numpy as np 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns 

import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import transforms
from dataloader import ImagenetteLoader
from utils import AverageMeter, LinearWarmupCosineAnnealingLR, get_val_transforms, get_train_transforms, args
from main import train, validate, save_checkpoint

from models import default_cnn_model

folds = -1
base = "bottleneck_classifier"
data_base = "../data/imagewoof2"

best_wd, best_lr = '0.001', '0.1'
model_folder = f"../models/best_{base}_lr{best_lr}_wd{best_wd}"

model = default_cnn_model(num_classes=10)
model_path = f"{model_folder}/model_best.pth.tar"
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint["state_dict"])
model = model.cuda()

args.name = "fine_tuned"
best_acc1 = -1
model_arch = "bottleneck"
args.print_freq = math.inf

# Freeze layers and replace head
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, 10)
model = model.cuda()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transforms = get_train_transforms(normalize)
val_transforms = get_val_transforms(normalize)

train_dataset = ImagenetteLoader("../data/imagewoof2", split="train", transforms=train_transforms, folds=-1, k=0)
val_dataset = ImagenetteLoader("../data/imagewoof2", split="test", transforms=val_transforms, folds=-1, k=0)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    pin_memory=True,
    sampler=None,
    drop_last=True,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    sampler=None,
    drop_last=False,
)

cudnn.enabled = True
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()

epochs = 60
args.epochs = epochs
freeze_epochs = 8

optimizer = torch.optim.SGD(
    model.parameters(),
    float(best_lr),
    momentum=0.9,
    weight_decay=float(best_wd),
)
scheduler = LinearWarmupCosineAnnealingLR(optimizer, 1, freeze_epochs * len(train_loader))

train_accs, losses = [], []
val_accs = []

print("Training the model ...")
for epoch in range(args.start_epoch, args.epochs):
    if epoch == freeze_epochs:
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.SGD(
            model.parameters(),
            float(best_lr)/5,
            momentum=0.9,
            weight_decay=float(best_wd),
        )
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, 5 * len(train_loader), (epochs - freeze_epochs) * len(train_loader))
    curr_train_accs, curr_losses = train(train_loader, model, criterion, optimizer, scheduler, epoch, args)
    train_accs += [curr_train_accs]
    losses += curr_losses
    curr_val_accs = validate(val_loader, model, epoch, args)
    acc1 = curr_val_accs.loc[0, 'val_top1']
    val_accs += [curr_val_accs]
    # remember best acc@1 and save checkpoint
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
