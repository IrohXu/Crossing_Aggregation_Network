#!/bin/bash

mkdir ./logs
mkdir ./models
dataset='gland_dataset'
for model in 'ca1' 'ca2' 'ca' 'unet' 'unetpp'
do
    for seed in {1..2}
    do
        python3 main.py --seed $seed --model-type $model --dataset $dataset | tee "./logs/$model.$dataset.seed$seed.log"
		wait
    done
done
