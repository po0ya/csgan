#!/usr/bin/env bash
set -x
MS="8,39,78,196,10,50,100,200"
# MNIST CSGAN with no contrastive loss regularizer on the latent space.
./experiments/scripts/diff_measurements_base_all.sh ./experiments/cfgs/csgan/csgan_mnist_disc.yml $MS