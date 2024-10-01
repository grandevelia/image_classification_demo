folds=5
base="bottleneck_classifier"

for wd in 0.00001 0.0001 0.001; do
    for lr in 0.001 0.01 0.1 1; do
        for ((k=0;k<$folds;k++)); do
            python main.py ../data/imagenette2 ${base}_lr${lr}_wd${wd}_k${k} --folds $folds --fold-index $k --lr $lr --wd $wd &
        done
        wait
    done
done


# Using best hyperparameters found using vis_cross_validation.py
best_lr=0.1
best_wd=0.001
python main.py ../data/imagenette2 best_${base}_lr${best_lr}_wd${best_wd} --lr $best_lr --wd $best_wd

# Now train only on imagewoof with the same params
python main.py ../data/imagewoof2 best_${base}_imagewoof_lr${best_lr}_wd${best_wd} --lr $best_lr --wd $best_wd

# Finally, fine-tune the best imagenette model on imagewoof
python fine_tune.py

