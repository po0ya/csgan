## Run scripts

This directory contains the scripts for all of the experiments in the paper. These experiments should be run from <proj_root>.
To reproduce the results reported in the paper refer to [experiments/scripts](experiments/scripts) which contains one script per results Table:

### Notes:
- These experiments should be run from <proj_root>.
-`generate_As.sh` should be run first, else the other experiments will not work.
- Each script runs its experiments in background. `stdout` and `stderr` outputs of each run are redirected to a file `debug/<cfg>/<exp_file>`.

### Scripts

- `generate_As.sh` will generate fixed random measuring matrices and saves them into `<proj_root>/output/sampling_mats/`. Usage:

        ./experiments/scripts/generate_As.sh

- `diff_measurements_base.sh` Trains and tests models with different number of measurements.

        ./experiments/scripts/diff_measurements_base.sh <path-to-cfg> <comma-separated number of measurements (10,20,100)> <extra configs (optional)>

- `figure_1_{mnist|fmnist|celeba}_reconstruction.sh` runs the experiments for Figure 1 using `diff_measurements_base.sh`.

        ./experiments/scripts/figure_1_{mnist|fmnist|celeba}_reconstruction.sh

- `table_1_base.sh` Trains and tests models with different number of seen uncompressed training data

        ./experiments/scripts/figure_1_base.sh <path-to-cfg> <comma-separated number of samples (100,1000,8000)> <extra configs (optional)>

- `table_1_{csgan|dcgan}.sh` runs the experiments of Table 1 using `table_1_base.sh`.

        ./experiments/scripts/table_1_{csgan|dcgan}.sh

- `table_2_{random|superres}.sh` runs the experiments of Table 2 using `table_1_base.sh`.

        ./experiments/scripts/table_2_{random|superres}.sh

- `table_3.sh` runs the experiments of Table 3.

        ./experiments/scripts/table_3.sh

- `diff_measurements_base.sh` Trains and tests models with different number of measurements, and extracts all features of {train|dev|test} sets.

        ./experiments/scripts/diff_measurements_base_all.sh <path-to-cfg> <comma-separated number of measurements (10,20,100)> <extra configs (optional)>

- `table_4_5_{mnist|fmnist}.sh` runs the experiments for training models with discriminative latent space of Table 4 and 5.

        ./experiments/scripts/table_4_5_{mnist|fmnist}.sh

- `cl_base.sh` The base script for classifying the saved features of each experiment. The path of the saved training features with `table_4_5_{mnist|fmnist}.sh` should be provided.

        ./experiments/scripts/cl_base.sh <classifier config> <path-to-train-features> --validate

- TODO: Table 4 and 5 results may not match the paper numbers, since TF seed was not fixed at the time of submission.
