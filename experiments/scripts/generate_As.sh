#!/usr/bin/env bash
set -x
set -e
CFG=experiments/cfgs/csgan/dcgan_mnist.yml
MS=(8 39 78 196 10 20 50 100 200 400)
for m in "${MS[@]}"
do
    python main.py --cfg $CFG  --cs_num_measurements $m --generate_A
done

CFG=experiments/cfgs/csgan/dcgan_fmnist.yml
MS=(10 20 50 100 200 400)
for m in "${MS[@]}"
do
    python main.py --cfg $CFG  --cs_num_measurements $m --generate_A
done


CFG=experiments/cfgs/csgan/dcgan_celeba.yml
MS=(20 50 100 200 500)
for m in "${MS[@]}"
do
    python main.py --cfg $CFG  --cs_num_measurements $m --generate_A
done