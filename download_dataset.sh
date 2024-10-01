#!/bin/bash
# script for downloading the dataset
mkdir -p data
cd data

wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
wget https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz

tar -xzf imagenette2.tgz
tar -xzf imagewoof2.tgz

# setup test using half of validation
mkdir -p imagenette2/test
mkdir -p imagewoof2/test

for dir in imagenette2/val/*/; do
    dir=$(basename $dir)
    echo $dir
    mkdir -p imagenette2/test/${dir}
    n_imgs=$(ll imagenette2/val/${dir} | wc -l)
    n_test=$(( $n_imgs / 2 ))
    i=0
    for img in imagenette2/val/${dir}/*; do
        fn=$(basename $img)
        mv imagenette2/val/${dir}/${fn} imagenette2/test/${dir}/${fn}
        i=$((i+1))
        if [[ $i == $n_test ]]; then
            break
        fi
    done
done

for dir in imagewoof2/val/*/; do
    dir=$(basename $dir)
    echo $dir
    mkdir -p imagewoof2/test/${dir}
    n_imgs=$(ll imagewoof2/val/${dir} | wc -l)
    n_test=$(( $n_imgs / 2 ))
    i=0
    for img in imagewoof2/val/${dir}/*; do
        fn=$(basename $img)
        mv imagewoof2/val/${dir}/${fn} imagewoof2/test/${dir}/${fn}
        i=$((i+1))
        if [[ $i == $n_test ]]; then
            break
        fi
    done
done

cd ..
