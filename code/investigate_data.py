import gnureadline
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import transforms
from transforms import *
from utils import load_image

fn = "../data/imagenette2/train/n02102040/ILSVRC2012_val_00000665.JPEG" 
img = load_image(fn)

# create an empty list and add transforms one by one
scale_transform = Compose([Scale(320)])(img)
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalize_transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(img)
rc_transform = Compose([RandomColor(0.15)])(img)
rsc_transform = Compose([RandomSizedCrop(224)])(img)

# let's take a look at the results!
plt.figure(figsize=(4,4))
plt.imshow(scale_transform)
plt.title("Rescale")
plt.savefig("../results/scale_img.png", bbox_inches='tight')

rhf_transform = Compose([RandomHorizontalFlip()])(img)
plt.figure(figsize=(4,4))
plt.imshow(rhf_transform)
plt.title("Random Horizontal Flip")
plt.savefig("../results/rhf_img.png", bbox_inches='tight')

plt.figure(figsize=(4,4))
plt.imshow(rc_transform)
plt.title("Random Color")
plt.savefig("../results/color_img.png", bbox_inches='tight')

rr_transform = Compose([RandomRotate(30)])(img)
plt.figure(figsize=(4,4))
plt.imshow(rr_transform)
plt.title("Random Rotate")
plt.savefig("../results/rotate_img.png", bbox_inches='tight')

plt.figure(figsize=(4,4))
plt.imshow(rsc_transform)
plt.title("Random Sized Crop")
plt.savefig("../results/random_crop_img.png", bbox_inches='tight')

plt.figure(figsize=(4,4))
plt.imshow(normalize_transform.transpose(0, 2).transpose(0, 1))
plt.title("Normalize")
plt.savefig("../results/normalize_img.png", bbox_inches='tight')