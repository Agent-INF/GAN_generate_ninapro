#!/bin/bash
for i in $(seq 1 9)
do
    python gan_nina_multi.py --iwgan False --dataname "00${i}"
    python gan_nina_bin00.py --iwgan False --is_test True --batch_size 100 --epoch 10 --dataname "00${i}"
done
for i in $(seq 10 52)
do
    python gan_nina_multi.py --iwgan False --dataname "0${i}"
    python gan_nina_bin00.py --iwgan False --is_test True --batch_size 100 --epoch 10 --dataname "0${i}"
done