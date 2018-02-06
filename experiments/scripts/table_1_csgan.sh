#!/usr/bin/env bash
set -x
OS="0,100,1000,8000"
# CelebA CSGAN
./experiments/scripts/table_1_base.sh ./experiments/cfgs/csgan/csgan_mnist_noncompressed.yml 100 $OS