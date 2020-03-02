#!/bin/bash

rm -r ./logs
mkdir ./logs
for model in 'ca' 'unet'
do
    for seed in {1..10}
    do
        python3 main.py --seed $seed --model-type $model > "./logs/$model.seed$seed.log"
	wait
    done
done
