#!/usr/bin/env bash
set -x
MS="10,25,50,100,200,400"
# FMNIST CSGAN
./experiments/scripts/diff_measurements_base.sh ./experiments/cfgs/csgan/csgan_fmnist.yml $MS

# FMNIST DCGAN
./experiments/scripts/diff_measurements_base.sh ./experiments/cfgs/csgan/dcgan_fmnist.yml $MS