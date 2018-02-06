#!/usr/bin/env bash
set -x
MS="10,25,50,100,200,400"
# MNIST CSGAN
./experiments/scripts/diff_measurements_base.sh ./experiments/cfgs/csgan/csgan_mnist.yml $MS

# MNIST DCGAN
./experiments/scripts/diff_measurements_base.sh ./experiments/cfgs/csgan/dcgan_mnist.yml $MS