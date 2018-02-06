#!/usr/bin/env bash
set -x
OS="100,1000,8000"
# MNIST DCGAN
./experiments/scripts/table_1_base.sh ./experiments/cfgs/csgan/dcgan_mnist_noncompressed.yml 100 $OS