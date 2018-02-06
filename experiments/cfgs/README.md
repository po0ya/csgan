## Configuration files

This directory contains the configuration files that are used in the paper:

- `csgan/csgan_{mnist|fmnist|celeba}.yml` contains the configuration files for CSGAN reconstruction experiments. (Algorithm 1 and Figure 1)
- `csgan/dcgan_{mnist|fmnist|celeba}.yml` contains the configuration files for DCGAN reconstruction experiments. (Figure 1)
- `csgan/{cs|dc}gan_{mnist|fmnist|celeba}_noncompressed.yml` contains the configuration file for Algorithm 2. (Table 1, Table 2, Table 3 (with ORIG_SIZE: 0), Figure 2, and Figure 3)
- `csgan/{cs|dc}gan_celeba_noncompressed_superres.yml` contains the configuration file for training the GAN with low resolution data with Algorithm 2. (Table 2)
- `csgan/{cs|dc}gan_celeba_superres.yml` is for the super resolution experiments on CelebA with Algorithm 1. (Figure 4).
- `csgan/csgan_{fmnist|mnist}_disc.yml` is for making the latent space discriminative (Table 4, Table 5, Table 6).
- `cls/lenet_map_{mnhist|fmnist}_z20.yml` is used for the LeNet classification of the latent space features of the images (Table 4, Table 5, Table 6).
- `cls/knn_{mnist|fmnist}_k50.yml` is used for the K-NN classification of the latent space features of the images (Table 5, Table 6).
- `default_cfg.yml` is the default values of the hyper parameters in case they are not set.
- `csgan/csgan_celeba_inpaint.yml` is the configuration file for the inpainting experiments. [not included in the paper].
