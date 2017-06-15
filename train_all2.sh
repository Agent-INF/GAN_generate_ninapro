#!/bin/bash
for i in $(seq 6 9)
do
    python gan_nina_multi.py --iwgan True --dataname "00${i}"
    python gan_nina_multi.py --iwgan True --is_test True --batch_size 100 --epoch 10 --dataname "00${i}"
done
for i in $(seq 10)
do
    python gan_nina_multi.py --iwgan True --dataname "0${i}"
    python gan_nina_multi.py --iwgan True --is_test True --batch_size 100 --epoch 10 --dataname "0${i}"
done