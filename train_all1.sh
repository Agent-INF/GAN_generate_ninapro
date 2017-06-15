#!/bin/bash
for i in $(seq 1 5)
do
    python gan_nina_multi.py --iwgan True --dataname "00${i}"
    python gan_nina_multi.py --iwgan True --is_test True --batch_size 100 --epoch 10 --dataname "00${i}"
done
