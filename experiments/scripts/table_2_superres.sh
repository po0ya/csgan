#!/usr/bin/env bash
set -x
OS="1000,4000,32000"
# CelebA CSGAN with limited number of high resolution images
./experiments/scripts/table_1_base.sh ./experiments/cfgs/csgan/csgan_celeba_noncompressed_superres.yml 500 $OS

# DCGAN CSGAN with limited number of high resolution images
./experiments/scripts/table_1_base.sh ./experiments/cfgs/csgan/dcgan_celeba_noncompressed_superres.yml 500 $OS