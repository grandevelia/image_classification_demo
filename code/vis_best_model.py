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
from utils import AverageMeter, LinearWarmupCosineAnnealingLR, get_val_transforms, get_train_transforms

from models import default_cnn_model

folds = 5
base = "bottleneck_classifier"
data_base = "../data/imagenette2"

best_wd, best_lr = '0.001', '0.1'
model_folder = f"../models/best_{base}_lr{best_lr}_wd{best_wd}"

################################################################################
# Training plots
################################################################################

losses = pd.read_csv(f"{model_folder}/losses.tsv", sep="\t")
val_acc = pd.read_csv(f"{model_folder}/validation_accuracy.tsv", sep="\t")
train_acc = pd.read_csv(f"{model_folder}/train_accuracy.tsv", sep="\t")

plt.figure()
plt.plot(train_acc['epoch'], train_acc['top1'], label='train')
plt.plot(val_acc['epoch'], val_acc['val_top1'], label='test')
plt.ylabel("accuracy")
plt.legend()
plt.savefig("../results/bottleneck/best_train_accuracy.png", bbox_inches="tight")
plt.close()

plt.figure()
losses = losses.rename(columns={"i": "epoch"})
plt.plot(losses['epoch'], losses['train_loss'])
plt.ylabel("training loss")
plt.savefig("../results/bottleneck/best_train_loss.png", bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(losses['epoch'], losses['lr'])
plt.ylabel("Learning Rate")
plt.savefig("../results/bottleneck/best_train_lr.png", bbox_inches="tight")
plt.close()


################################################################################
# load model, get sensitivity, specificity, confusion matrix
################################################################################

model = default_cnn_model(num_classes=10)
model_path = f"{model_folder}/model_best.pth.tar"
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint["state_dict"])
model = model.cuda()
model.eval()

cudnn.deterministic = True

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_transforms = get_val_transforms(normalize)
val_dataset = ImagenetteLoader("../data/imagenette2", split="test", transforms=val_transforms, folds=-1, k=0)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    sampler=None,
    drop_last=False,
)

pred, true = [], []
for i, (input, target) in enumerate(val_loader):
    input = input.cuda(non_blocking=False)
    true += [int(target.numpy())]
    pred += [np.argmax(model(input).detach().cpu().numpy())]

conf_mat = confusion_matrix(true, pred)
conf_mat = pd.DataFrame(conf_mat)
index_map = val_dataset.get_index_mapping()
cats = list(index_map.values())
conf_mat.index, conf_mat.columns = cats, cats

plt.figure()
sns.heatmap(conf_mat, cmap="viridis", annot=True, fmt='d')
plt.savefig("../results/bottleneck/confusion_matrix.png", bbox_inches="tight")

conf_mat = np.array(conf_mat)
sens_mat = np.zeros(shape=conf_mat.shape[0])
for i in range(conf_mat.shape[0]):
    tp = conf_mat[i, i]
    fn = conf_mat[i, :].sum() - tp
    sens_mat[i] = tp / (tp + fn)

plt.figure()
sens_mat = pd.DataFrame(sens_mat)
sens_mat.index = cats
sns.heatmap(sens_mat, cmap="viridis", annot=True)
plt.savefig("../results/bottleneck/sensitivities.png", bbox_inches="tight")
print("Average Sensitivity", np.array(sens_mat).mean()) # 0.326


spec_mat = np.zeros(shape=conf_mat.shape[0])
for i in range(conf_mat.shape[0]):
    tn = np.diag(conf_mat).sum() - conf_mat[i, i]
    fp = conf_mat[:, i].sum() - conf_mat[i, i]
    spec_mat[i] = tn / (tn + fp)


plt.figure()
spec_mat = pd.DataFrame(spec_mat)
spec_mat.index = cats
sns.heatmap(spec_mat, cmap="viridis", annot=True)
plt.savefig("../results/bottleneck/specificities.png", bbox_inches="tight")
print("Average Specificity", np.array(spec_mat).mean()) # 0.852

