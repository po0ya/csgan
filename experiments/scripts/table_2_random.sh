#!/usr/bin/env bash
set -x
OS="1000,4000,32000"
# CelebA CSGAN
./experiments/scripts/table_1_base.sh ./experiments/cfgs/csgan/csgan_celeba_noncompressed.yml 500 $OS

# CelebA DCGAN
./experiments/scripts/table_1_base.sh ./experiments/cfgs/csgan/dcgan_celeba_noncompressed.yml 500 $OS