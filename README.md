### Classifying Imagenette with a small bottleneck network

This project outlines the workflow for training a small classifier on [Imagenette](https://github.com/fastai/imagenette). The general workflow is as follows:


1. __Download the datasets and set up__
```
bash download_dataset.sh
pip install -r requirements.txt
```
2. __Perform 5-fold cross-validation on learning rate and L2 Regularization strength__
```
cd code
bash train.sh
```
3. __Using the optimal hyperparameters discovered in step 2, train a model on the full training set, e.g.__
```
base=bottleneck
best_lr=0.1
best_wd=0.001
python main.py ../data/imagenette2 best_${base}_lr${best_lr}_wd${best_wd} --lr $best_lr --wd $best_wd
```

4. __Train a second model on Imagewoof alone__
```
python main.py ../data/imagewoof2 best_${base}_imagewoof_lr${best_lr}_wd${best_wd} --lr $best_lr --wd $best_wd
```

5. __Fine tune the original model on Imagewoof to see if transfer learning can improve performance on this more-challening dataset__
```
python fine_tune.py
```
