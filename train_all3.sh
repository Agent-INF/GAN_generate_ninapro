#!/bin/bash
for i in $(seq 11 15)
do
    python gan_nina_multi.py --gpu 0 --iwgan True --dataname "0${i}"
    python gan_nina_multi.py --gpu 0 --iwgan True --is_test True --batch_size 100 --epoch 10 --dataname "0${i}"
done