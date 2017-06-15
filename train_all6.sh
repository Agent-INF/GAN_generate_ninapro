#!/bin/bash
for i in $(seq 26 30)
do
    python gan_nina_multi.py --gpu 1 --iwgan True --dataname "0${i}"
    python gan_nina_multi.py --gpu 1 --iwgan True --is_test True --batch_size 100 --epoch 10 --dataname "0${i}"
done