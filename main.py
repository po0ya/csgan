# Copyright 2018 The CSGAN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Where CSGAN models are trained and used for reconstructing compressed samples."""

import argparse
import sys
import os
import tensorflow as tf
import numpy as np

from datasets.factory import load_ds
from utils.config import load_config, set_default_params
from utils.csgan_utils import generate_A

FLAGS = tf.app.flags.FLAGS


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', required=True, help='Config file')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args, _ = parser.parse_known_args()
    return args


def main(cfg, *args):
    # Initialization of the rest of FLAG variables
    # Set seeds for reproducing the results
    tf.set_random_seed(1234)
    np.random.seed(1234)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height
    tf.app.flags.DEFINE_integer('n_inputs',FLAGS.output_width *
                                FLAGS.output_height * \
                          FLAGS.c_dim,'Dimension of the original image.')

    # These imports are here because FLAGS values are not initialized correctly before here
    from models.csgan import CSGAN
    from utils.csgan_utils import save_mse
    from utils.csgan_utils import save_x_hats

    # Creating the directories
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):  # This shouldn't be needed
        os.makedirs(FLAGS.sample_dir)

    # DEFAULT SUPERRES FACTOR IS SET TO 4, should be moved to default cfg later
    if FLAGS.dc_super_res:
        assert not FLAGS.cs_learning
        tf.app.flags.DEFINE_integer('superres_factor',4,'Super resolution '
                                                        'factor')
        FLAGS.superres_factor = 4

    # Generate the sampling matrix suitable for the CSGAN object and quit.
    if FLAGS.generate_A:
        generate_A()
        exit()

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:

        csgan = CSGAN(sess,
                      input_height=FLAGS.input_height,
                      input_width=FLAGS.input_width,
                      is_crop=FLAGS.is_crop,
                      batch_size=FLAGS.batch_size,
                      sample_num=FLAGS.batch_size,
                      output_height=FLAGS.output_height,
                      output_width=FLAGS.output_width,
                      z_dim=FLAGS.z_dim,
                      c_dim=FLAGS.c_dim,
                      dataset_name=FLAGS.dataset,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      cfg=cfg)


        # Train.
        if FLAGS.is_train:
            # Initialize from a saved weight.
            if FLAGS.initialize_from is not None:
                # Priority is for the default path of the csgan object. If it has been trained already, load that model.
                could_load, counter = csgan.load()

                if not could_load and not FLAGS.retrain:
                    csgan.load_from_path(FLAGS.initialize_from)
                    csgan.counter = 0
                    print('[#] Pre-initializing with {}'.format(
                        FLAGS.initialize_from))
                else:
                    print('[#] Loaded the saved weights')
            # Train the model.
            csgan.train()

        else:
            # In test phase, just load the weights.
            could_load, counter = csgan.load()
            if not could_load:
                raise Exception("[!] Train a model first, then run test mode.")

        # Set random restart from flags.
        if FLAGS.test_rr != -1:
            FLAGS.cs_num_random_restarts = FLAGS.test_rr

        # PREDEFINED NUM_RR UPDATE_ITER FOR CELEBA AND MNIST [default values used in the paper].
        if FLAGS.default_test_params:
            set_default_params()

        # Loads the dataset object.
        ds = load_ds()

        if FLAGS.reconstruction_res:  # Get reconstruction results.
            if FLAGS.extract_all_feats or (
                    FLAGS.is_train and FLAGS.test_results_split == 'train'):
                for s in ['train', 'test', 'val']:
                    FLAGS.test_results_split = s
                    save_mse(csgan, ds)
            else:
                save_mse(csgan, ds)
        if FLAGS.save_xs:
            save_x_hats(csgan, ds)


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args.cfg)

    # Set derived configs and never experiment independent configs.
    flags = tf.app.flags
    flags.DEFINE_boolean("is_train", False,
                         "True for training, False for testing [False]")
    flags.DEFINE_boolean("debug", False, "Debug mode [False]")
    flags.DEFINE_string("test_results_split", "train",
                        "The split to save test results on [train]")
    flags.DEFINE_boolean("test_recompute", False,
                         "Re-compute test results [False]")
    flags.DEFINE_integer("test_batch_size", 20, "Test batch size [20]")
    flags.DEFINE_integer("cs_decay_lr_iter", 1000, "Step of learning rate "
                                                   "decay [1000]")
    flags.DEFINE_integer("num_tests", -1,
                         "Number of tests after test_id flag [10]")
    flags.DEFINE_string("sampling_mat_dir", 'output/sampling_mats',
                        'The directory containing sampling matrices [outputs/sampling_mats]')
    flags.DEFINE_boolean("vis", False, "Visualize the results [False]")
    flags.DEFINE_boolean("reconstruction_res", False,
                         "Save mse of reconstruction [False]")
    flags.DEFINE_boolean("tensorboard_log", False,
                         "Save tensorboard log [False]")
    flags.DEFINE_integer("test_id", -1,
                         "Only test on the given test_id, -1 tests on all ["
                         "-1]")
    flags.DEFINE_integer('a_ind', 0, "The index of saved sampling matrix")
    flags.DEFINE_integer('k_h', 5, "Convolution kernel height")
    flags.DEFINE_integer('k_w', 5, "Convolution kernel width")
    flags.DEFINE_string('ckpt_path', "", "A custom checkpoint path")
    flags.DEFINE_string('output_dir', "output/default",
                        "The root directory of where checkpoints live.")
    flags.DEFINE_integer('test_rr', -1, "Number of random restarts, "
                                        "a shorter version of "
                                        "--cs_num_random_restarts")
    flags.DEFINE_boolean('default_test_params', False,
                         "Use the test parameters that are used in the "
                         "paper.")
    flags.DEFINE_boolean('save_xs', False, 'Save images of x hats')
    flags.DEFINE_boolean('retrain', False, 'Re-train the model from scratch')
    flags.DEFINE_boolean('generate_A', False, 'Generates measurement matrices')
    flags.DEFINE_boolean('keep_all', False, 'Keep all the checkpoints')
    flags.DEFINE_boolean('extract_all_feats', False,
                         'Extract features for all the splits.')
    flags.DEFINE_boolean('dc_inpaint', False, 'DCGAN inpainting testing')
    flags.DEFINE_boolean('dc_super_res', False, 'DCGAN super resolution '
                                                'testing')
    flags.DEFINE_boolean('prior_test', False, 'Initialize the zs with a '
                                              'prior value that comes from '
                                              'upsampling. Should be used '
                                              'with --superres_factor')
    flags.DEFINE_float('dc_inpaint_ratio', 0, 'DCGAN ratio of inpainting '
                                              'window.')
    flags.DEFINE_string('initialize_from', None, 'Initialize the GAN from a '
                                                 'checkpoint before training.')

    main_cfg = lambda x: main(cfg, x)
    tf.app.run(main=main_cfg)
