#!/bin/bash
for i in $(seq 1 9)
do
    #python gan_nina_bin00.py --gpu 0 --iwgan False --dataname "00${i}"
    python gan_nina_bin00.py --gpu 0 --iwgan False --is_test True --batch_size 400 --epoch 10 --dataname "00${i}"
done
for i in $(seq 10 13)
do
    #python gan_nina_bin00.py --gpu 0 --iwgan False --dataname "0${i}"
    python gan_nina_bin00.py --gpu 0 --iwgan False --is_test True --batch_size 400 --epoch 10 --dataname "0${i}"
done