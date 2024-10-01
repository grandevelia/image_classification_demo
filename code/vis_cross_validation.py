import gnureadline
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

folds = 5
base = "bottleneck_classifier"
data_base = "../data/imagenette2"

best_wd, best_lr, best_acc = -1, -1, -1

dat = []
for wd in ["0.00001", "0.0001", "0.001"]:
	for lr in ["0.001", "0.01", "0.1", "1"]:
		total_acc = 0
		for k in range(folds):
			out_dir = f"../models/{base}_lr{lr}_wd{wd}_k{k}"
			total_acc += pd.read_csv(f"{out_dir}/validation_accuracy.tsv", sep="\t")['val_top1'].max()
		curr_acc = total_acc / folds
		dat += [pd.DataFrame(data=[{"wd": wd, "lr": lr, "acc": curr_acc}])]
		if curr_acc > best_acc:
			best_acc = curr_acc
			best_wd, best_lr = wd, lr

dat = pd.concat(dat)
dat = dat.astype(float)

yticks = dat['lr'].unique()
xticks = dat['wd'].unique()

y_map = {y: i for i, y in enumerate(yticks)}
x_map = {x: i for i, x in enumerate(xticks)}

dat['lr'] = [y_map[y] for y in dat['lr']]
dat['wd'] = [x_map[x] for x in dat['wd']]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
colors = plt.cm.viridis(dat['acc']/float(dat['acc'].max()))
bottom = np.zeros_like(dat['lr'])
surf = ax.bar3d(dat['wd'], dat['lr'], bottom, 1, 1, dat['acc'], color=colors, shade=True, linewidth=0.2)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set(xlabel='Weight Decay', ylabel='Learning Rate')
ax.view_init(30, 30, 0)
ax.set_xticks(dat['wd'].unique(), labels=xticks)
ax.set_yticks(dat['lr'].unique(), labels=yticks)
plt.show()
plt.savefig("../bn_cross_validation.png", bbox_inches="tight")
plt.close()
