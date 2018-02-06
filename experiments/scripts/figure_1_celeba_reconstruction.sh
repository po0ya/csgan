#!/usr/bin/env bash
set -x
MS="20,50,100,200,500"
# CelebA CSGAN
./experiments/scripts/diff_measurements_base.sh ./experiments/cfgs/csgan/csgan_celeba.yml $MS

# CelebA DCGAN
./experiments/scripts/diff_measurements_base.sh ./experiments/cfgs/csgan/dcgan_celeba.yml $MS