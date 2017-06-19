#!/bin/bash
for i in $(seq 40 52)
do
    #python gan_nina_bin00.py --gpu 3 --iwgan False --dataname "0${i}"
    python gan_nina_bin00.py --gpu 3 --iwgan False --is_test True --batch_size 400 --epoch 10 --dataname "0${i}"
done