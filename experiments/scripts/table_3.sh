#!/usr/bin/env bash
set -x
MS="25,50,100,200"
# MNIST CSGAN with no seen uncompressed data
./experiments/scripts/diff_measurements_base.sh ./experiments/cfgs/csgan/csgan_mnist_noncompressed.yml $MS --orig_size 0

# FMNIST CSGAN with no seen uncompressed data
./experiments/scripts/diff_measurements_base.sh ./experiments/cfgs/csgan/csgan_fmnist_noncompressed.yml $MS --orig_size 0