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

"""Contains the implementation of the CSGAN class."""

from __future__ import division

import numpy as np
import os
import re
import cPickle
import time
import yaml
from pip.utils import ensure_dir

from main import load_ds
from utils.csgan_utils import get_cs_learning_rate, get_A_superres, DummyWriter, \
    get_A_inpaint, get_A_path
from ops import *

FLAGS = tf.app.flags.FLAGS


class CSGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, is_crop=True, batch_size=64, sample_num=64,
                 output_height=64, output_width=64, z_dim=100, gf_dim=64, df_dim=64, gfc_dim=1024,
                 dfc_dim=1024, c_dim=3, dataset_name='default', checkpoint_dir=None, cfg=None):
        """The constructor of CSGAN class. An object containing a computation graph corresponding to a loaded config is
        created.

        Args:
          sess: TensorFlow session
          batch_size: The batch size. Should be specified before training.
          z_dim: (optional) Dimension of the latent space (z). [100]
          gf_dim: (optional) Dimension of generator filters in the first convolutional layer. [64]
          df_dim: (optional) Dimension of discriminator filters in the first convolutional layer. [64]
          gfc_dim: (optional) Dimension of generator units for the fully-connected layer. [1024]
          dfc_dim: (optional) Dimension of discriminator units for the fully-connected layer. [1024]
          c_dim: (optional) Number of image channels. For grayscale input, set to 1. [3]
        """
        self.cache_path = 'data/cache'
        ensure_dir(self.cache_path)

        self.cfg = cfg
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim  # Latent dimension.

        self.gf_dim = gf_dim  # Generator convolutional dimension.
        self.df_dim = df_dim  # Discriminator convolutional dimension.

        self.gfc_dim = gfc_dim  # Generator fully-connected dimension.
        self.dfc_dim = dfc_dim  # Discriminator fully-connected dimension.

        self.c_dim = c_dim  # Number of channels.

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir

        # DCGAN or CSGAN.
        self.cs_learning = False  # If we are in the CS learning mode.
        self.cs_m_and_orig = False
        # If we are training on measurements (ms) and uncompressed (orig)
        # samples, refer to Algorithm 2 of the paper.

        if hasattr(FLAGS, 'cs_learning'):  # Setting CSGAN flag.
            if FLAGS.cs_learning:
                self.cs_learning = True
            if hasattr(FLAGS, 'orig_size') and self.cs_learning:
                if FLAGS.orig_size > 0:
                    self.cs_m_and_orig = True
                if FLAGS.orig_size == 0:
                    FLAGS.batch_size = 0

        # Contrastive loss will be added in loss().
        self.contrastive_learning = False
        if hasattr(FLAGS, 'cs_margin'):
            self.contrastive_learning = True

        # Super-resolution learning.
        self.superres_learning = False
        if hasattr(FLAGS, 'superres_factor'):
            self.superres_learning = True

        # Inpainting learning.
        self.inpaint_learning = False
        if hasattr(FLAGS, 'inpaint_ratio'):
            self.inpaint_learning = True

        self.keep_all = FLAGS.keep_all  # To keep all the model checkpoints.

        self.build_model()
        self._save_cfg_in_ckpt()

    def _save_cfg_in_ckpt(self):
        """Saves the configuration of the object into a file to facilitate future loads.

        This function will create an output directory based on the config
        file and the flags. It will use the filename of the config file as
        its directory name. If the flags are different from what is set in
        the config file, the name of the directory will be extended to contain
        those flags and their values. It will also create a new config file
        and saves it in the experiment directory so that in the future
        loadings of this model, that config file is read.
        """

        final_cfg = {}

        self.model_postfix = ''

        if hasattr(self, 'cfg'):
            if self.cfg is not None:
                flags_dict = {k: getattr(FLAGS, k) for k in FLAGS}  # Get the
                # TensorFlow flags.
                # If the filename is cfg.yml then just set ckpt_dir to the the config path since it is the same as
                # where checkpoint files are saved.
                if 'cfg.yml' in flags_dict['cfg_path']:
                    self.ckpt_dir = os.path.dirname(flags_dict['cfg_path'])
                else:
                    for attr in flags_dict.keys():
                        if attr.upper() in self.cfg.keys():
                            self_val = flags_dict[attr]
                            if self_val is not None:
                                if self_val != self.cfg[attr.upper()]['val']:
                                    final_cfg[attr.upper()] = {'val': self_val, 'doc': self.cfg[attr.upper()]['doc']}
                                    if not (attr == 'zprior_weight' or attr == 'dataset' or attr == 'cfg_file' or
                                            'width' in attr or 'height' in attr):
                                        self.model_postfix += 'II{}XX{}'.format(attr, self_val).replace('.', '_')
                                else:
                                    final_cfg[attr.upper()] = self.cfg[attr.upper()]

                    self.ckpt_dir = os.path.join(self.checkpoint_dir, self.model_dir)
                    if FLAGS.is_train:
                        ensure_dir(self.ckpt_dir)
                        with open(os.path.join(self.checkpoint_dir, self.model_dir, 'cfg.yml'), 'w') as f:
                            yaml.dump(final_cfg, f)

    def build_model(self):
        """Creates the appropriate TensorFlow computation graph."""

        # If center crop is valid, output dimensions will be the image dimensions.

        if self.is_crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            # Otherwise, input dimensions will not be changed.
            image_dims = [self.input_height, self.input_width, self.c_dim]
        self.image_dims = image_dims

        # GAN latent variable.
        self.z = tf.placeholder(tf.float32, [FLAGS.batch_size + FLAGS.cs_batch_size, self.z_dim], name='z')

        if not FLAGS.just_cs:  # If we are not training only on measurements (non-compressed data is present).
            self.inputs = tf.placeholder(tf.float32, [
                FLAGS.batch_size + FLAGS.cs_batch_size] + image_dims, name='real_images')
            self.sample_inputs = tf.placeholder(tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

        if self.cs_learning:  # If not DCGAN.
            # Load the A (sampling) matrix.
            if self.superres_learning:
                self.A_val = get_A_superres()
                FLAGS.cs_num_measurements = self.A_val.shape[1]
            elif self.inpaint_learning:
                self.A_val = get_A_inpaint()
                FLAGS.cs_num_measurements = self.A_val.shape[1]
            else:
                # Use the same A matrix for all of the experiments.
                A_path = os.path.join(FLAGS.sampling_mat_dir,
                                      '{}_w{}_h{}_m{}_a{}.pckl'.format(FLAGS.dataset, FLAGS.output_width,
                                                                       FLAGS.output_height, FLAGS.cs_num_measurements,
                                                                       FLAGS.a_ind))
                if os.path.exists(A_path):
                    with open(A_path, 'r') as f:
                        self.A_val = cPickle.load(f)
                else:
                    raise RuntimeError('[!] No A matrix was found at {}. Run the following command first:\n'
                                       'python main.py --cfg <config> --generate_A'.format(A_path))

            self.A = tf.constant(self.A_val, dtype=tf.float32)

            # Batch size for compressed measurements.
            cs_bsize = FLAGS.cs_batch_size
            if self.cs_m_and_orig:
                # In algorithm 2, we use the compressed original samples to
                # train the discriminator of the measurements too.
                cs_bsize = cs_bsize + FLAGS.batch_size

            if self.cs_m_and_orig:
                print('[#] Solving for both the measurements and original inputs.')
                # Placeholder for original samples.
                self.orig_inputs = tf.placeholder(tf.float32, [
                    FLAGS.batch_size] + image_dims, name='real_images')
            elif FLAGS.just_cs:
                print('[#] Solving for just the measurements, not the original inputs.')

            # Placeholder for compressed measurements.
            self.xs_target = tf.placeholder(tf.float32, [cs_bsize] + self.image_dims, name='target_xs')
            # Compute y_batch on GPU.
            self.y_batch = tf.matmul(tf.reshape(self.xs_target, [cs_bsize, -1]), self.A) + \
                           FLAGS.cs_noise_std * tf.random_normal([cs_bsize, FLAGS.cs_num_measurements])

            if FLAGS.just_cs:  # If we are in only compressed measurements mode, the inputs are y_batches.
                self.inputs = self.y_batch

        # Batch normalization (deals with poor initialization helps gradient flow).
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        # Generator network.
        self.G = self.generator(self.z, batch_size=FLAGS.cs_batch_size + FLAGS.batch_size)

        # Discriminator network outputs.
        self.D, self.D_logits = self.discriminator(self.inputs, batch_size=FLAGS.cs_batch_size + FLAGS.batch_size)

        def sigmoid_cross_entropy_with_logits(x, y):
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

        # If both compressed measurements and original images are used.
        if self.cs_m_and_orig:
            measured_generator_output = tf.matmul(
                tf.reshape(self.G, [FLAGS.cs_batch_size + FLAGS.batch_size, FLAGS.n_inputs]), self.A)
            self.D_, self.D_logits_ = self.discriminator(measured_generator_output, reuse=True,
                                                         batch_size=FLAGS.cs_batch_size + FLAGS.batch_size)
            # Create disciriminator for original inputs.
            self.D_orig, self.D_logits_orig = self.discriminator(self.orig_inputs, reuse=False,
                                                                 batch_size=FLAGS.batch_size, orig=True)
            # Give the generator to the original inputs discriminator.
            self.D_orig_, self.D_logits_orig_ = self.discriminator(self.G, reuse=True,
                                                                   batch_size=FLAGS.cs_batch_size + FLAGS.batch_size,
                                                                   orig=True)

            # Losses and summaries.
            self.d_loss_orig_real = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.D_logits_orig, tf.ones_like(self.D_orig)))

            self.d_loss_orig_fake = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.D_logits_orig_, tf.zeros_like(self.D_orig_)))

            self.d_loss_orig_sum = scalar_summary("d_loss_orig_sum", self.d_loss_orig_real + self.d_loss_orig_fake)

            self.g_loss_orig = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.D_logits_orig_, tf.ones_like(self.D_orig_)))

        # If only compressed measurements are used.
        elif FLAGS.just_cs:
            # The measurements on the output of the generator.
            measured_generator_output = tf.matmul(
                tf.reshape(self.G, [FLAGS.cs_batch_size + FLAGS.batch_size, FLAGS.n_inputs]), self.A)

            self.D_, self.D_logits_ = self.discriminator(measured_generator_output, reuse=True,
                                                         batch_size=FLAGS.cs_batch_size + FLAGS.batch_size)

        # If only original images are used.
        else:
            self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True,
                                                         batch_size=FLAGS.cs_batch_size + FLAGS.batch_size)

        # Losses and summaries.
        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))

        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        # If both compressed measurements and original images are used, add the two losses.
        if self.cs_m_and_orig:
            self.g_loss = self.g_loss + self.g_loss_orig
            self.d_loss = self.d_loss + self.d_loss_orig_real + self.d_loss_orig_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.global_variables()

        # Gather variables.
        self.d_vars = [var for var in t_vars if 'disc' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.other_vars = [var for var in tf.global_variables() if 'save_' in var.name]

        self.save_var_names = self.d_vars + self.g_vars + self.other_vars

        # Build the Compressed Sensing estimator.
        if self.cs_learning:
            self.cs_grad_estimator()

        # Keep all checkpoints.
        if self.keep_all:
            self.saver = tf.train.Saver(self.save_var_names, max_to_keep=None)
        else:
            self.saver = tf.train.Saver(self.save_var_names)

    def cs_loss(self, z_batch, cs_batch_size=None):
        """The compressed sensing loss over z.

        Args:
            z_batch: The Z tenosr.
            cs_batch_size: (optional) the batch size of
        """

        # Use the default batch size if it is not passed to the function.
        if cs_batch_size is None:
            cs_batch_size = FLAGS.cs_batch_size

        # Define the reconstructions.
        self.x_hat_batch = self.generator(z_batch, reuse=True, batch_size=cs_batch_size, train=False)

        # The compressed version of the reconstructions.
        y_hat_batch = tf.matmul(tf.reshape(self.x_hat_batch, [cs_batch_size, -1]), self.A, name='y2_batch')

        # Define reconstruction losses.
        # [begin] This part is based on https://github.com/AshishBora/csgm.
        self.m_loss1_batch = tf.reduce_mean(tf.abs(self.y_batch - y_hat_batch), 1)
        self.m_loss2_batch = tf.reduce_mean((self.y_batch - y_hat_batch) ** 2, 1)
        self.cs_total_loss_batch = FLAGS.mloss1_weight * self.m_loss1_batch \
                                   + FLAGS.mloss2_weight * self.m_loss2_batch \
                                   + FLAGS.zprior_weight * self.zp_loss_batch
        self.cs_total_loss = tf.reduce_mean(self.cs_total_loss_batch)
        self.m_loss1 = tf.reduce_mean(self.m_loss1_batch)
        self.m_loss2 = tf.reduce_mean(self.m_loss2_batch)
        self.zp_loss = tf.reduce_mean(self.zp_loss_batch)
        # [end] based on https://github.com/AshishBora/csgm.

        # Loss summaries.
        self.m_loss1_sum = scalar_summary('cs_m_loss1', self.m_loss1)
        self.m_loss2_sum = scalar_summary('cs_m_loss2', self.m_loss2)
        self.cs_z_sum = histogram_summary('cs_z_hist', z_batch)
        self.x_hat_sum = image_summary('x_hat_summ', self.x_hat_batch)
        self.cs_total_loss_sum = scalar_summary('cs_total_loss', self.cs_total_loss)
        self.cs_loss_rec_sum = scalar_summary('cs_loss_rec', tf.reduce_mean(self.cs_total_loss_batch))

        summaries = [self.m_loss1_sum, self.m_loss2_sum, self.cs_z_sum, self.cs_total_loss_sum, self.cs_loss_rec_sum,
                     self.x_hat_sum]

        # Define contrastive loss [can be readily replaced with tripplet loss, or center loss].
        self.cont_loss = tf.constant(0)
        if self.contrastive_learning:
            self.cont_loss = 0

            def euc_dist(x, y):
                d = tf.reduce_sum(tf.square(tf.subtract(x, y)))
                return d

            # Prepares the graph for contrastive loss.
            for i in range(FLAGS.cs_batch_size):
                for j in range(i + 1, FLAGS.cs_batch_size):
                    y = tf.cast(tf.equal(self.cs_input_labels[i], self.cs_input_labels[j]), tf.float32)
                    dist = euc_dist(z_batch[i, :], z_batch[j, :])
                    max_part = tf.maximum(FLAGS.cs_margin - dist, 0)
                    l = y * dist + (1 - y) * max_part
                    self.cont_loss = self.cont_loss + l

            self.cont_loss = (2.0 / (
                    FLAGS.cs_batch_size * (FLAGS.cs_batch_size - 1))) * self.cont_loss * FLAGS.cont_lmbd
            self.cont_loss_sum = scalar_summary("cont_loss", self.cont_loss)
            summaries.append(self.cont_loss_sum)

            # Add the contrastive loss to the reconstruction loss.
            self.cs_total_loss = self.cs_total_loss + self.cont_loss

        self.cs_sum = tf.summary.merge(summaries)

    def get_vars_by_prefix(self, prefix):
        """Gets all variable names with given prefix.

        Args:
            prefix: The prefix string to look for.

        Returns:
            TensorFlow variables with the specified prefix.
        """

        t_vars = tf.global_variables()
        return [var for var in t_vars if prefix in var.name]

    def discriminator(self, input, reuse=False, batch_size=None, orig=False, train=True):
        """ Builds the discriminator.
        Based on https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py

        Args:
             input: Tensors of real images or generator outputs.
             reuse: (optional) True at For test time.
             batch_size: (optional) Batch size.
             orig: (optional) To create a separate discriminator for original samples.
             train: (optional) True = training phase, False = test phase
        """

        # Preparing the discriminator name, since there might be two of them in the graph.
        if self.cs_m_and_orig and not orig:  # Discriminator of the compressed measurements.
            disc_name = "m_disc"
        else:
            disc_name = "discriminator"

        # Different discriminator architectures based on FLAGS.disc_type - reused from DCGAN.
        with tf.variable_scope(disc_name) as scope:
            if reuse:
                scope.reuse_variables()

            if FLAGS.just_cs and not orig:
                if FLAGS.disc_type == 'conv':
                    s_h16 = 32
                    s_w16 = 32
                    ch = 8
                    z_, self.h0_w, self.h0_b = linear(input, ch * s_h16 * s_w16, 'd_h0_lin', with_w=True)
                    h0 = tf.reshape(z_, [-1, s_h16, s_w16, ch])
                    input = tf.nn.relu(self.g_bn0(h0, train=train))

            if FLAGS.disc_type == 'conv':
                h0 = lrelu(conv2d(input, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, k_h=3, k_w=3, name='d_h1_conv'), train=train))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, k_h=3, k_w=3, name='d_h2_conv'), train=train))
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, k_h=3, k_w=3, name='d_h3_conv'), train=train))
                h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h3_lin')
            elif FLAGS.disc_type == 'fc':
                input = tf.reshape(input, [batch_size, -1])
                h0 = lrelu(self.d_bn1(linear(input, 500, "d_h0_fc"), train=train))
                h1 = lrelu(self.d_bn2(linear(h0, 500, "d_h1_fc"), train=train))
                h2 = lrelu(self.d_bn3(linear(h1, 500, "d_h2_fc"), train=train))
                h4 = linear(h2, 1, 'd_h3_lin')
            else:
                raise NotImplementedError

            return tf.nn.sigmoid(h4), h4

    def generator(self, z, reuse=False, batch_size=None, train=True):
        """This function builds the generator network. FLAGS.generator_type define different types of generators.
        Based on https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py

        Args:
            z: Tensor of latent vectors.
            reuse: Reuse for test time.
            batch_size: Batch size.
            train: Training phase, true: Training, false: Testing.

        Returns:
            The embeddings of the final layer.
        """

        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()

            if FLAGS.generator_type == 'conv':
                s_h, s_w = self.output_height, self.output_width

                def conv_out_size_same(size, stride):
                    return int(np.ceil(float(size) / float(stride)))

                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # Project z and reshape.
                self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

                self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(self.h0, train=train))

                self.h1, self.h1_w, self.h1_b = deconv2d(h0, [batch_size, s_h8, s_w8, self.gf_dim * 4], k_h=FLAGS.k_h,
                                                         k_w=FLAGS.k_w, name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1, train=train))

                h2, self.h2_w, self.h2_b = deconv2d(h1, [batch_size, s_h4, s_w4, self.gf_dim * 2], k_h=FLAGS.k_h,
                                                    k_w=FLAGS.k_w, name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2, train=train))

                h3, self.h3_w, self.h3_b = deconv2d(h2, [batch_size, s_h2, s_w2, self.gf_dim * 1], k_h=FLAGS.k_h,
                                                    k_w=FLAGS.k_w, name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3, train=train))
                h4, self.h4_w, self.h4_b = deconv2d(h3, [batch_size, s_h, s_w, self.c_dim], k_h=FLAGS.k_h,
                                                    k_w=FLAGS.k_w, name='g_h4', with_w=True)
                g = tf.nn.tanh(h4)

            elif FLAGS.generator_type == 'fc_dcgan':
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=train))
                h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin'), train=train))
                h1 = tf.reshape(h1, [batch_size, s_h4, s_w4, self.gf_dim * 2])

                h2 = tf.nn.relu(self.g_bn2(
                    deconv2d(h1, [batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=train))
                g = tf.nn.sigmoid(deconv2d(h2, [batch_size, s_h, s_w, self.c_dim], name='g_h3'))

            elif FLAGS.generator_type == 'fc_vae':
                # [begin] Based on https://github.com/AshishBora/mnist-vae.
                n_z = 20
                n_hidden_gener_1 = 500
                n_hidden_gener_2 = 500
                n_input = np.prod(FLAGS.image_shape)

                weights1 = tf.get_variable('g_w1', shape=[n_z, n_hidden_gener_1])
                bias1 = tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32), name='g_b1')
                hidden1 = tf.nn.softplus(tf.matmul(z, weights1) + bias1, name='g_h1')

                weights2 = tf.get_variable('g_w2', shape=[n_hidden_gener_1, n_hidden_gener_2])
                bias2 = tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32), name='g_b2')
                hidden2 = tf.nn.softplus(tf.matmul(hidden1, weights2) + bias2, name='g_h2')

                w_out = tf.get_variable('g_w_out', shape=[n_hidden_gener_2, n_input])
                b_out = tf.Variable(tf.zeros([n_input], dtype=tf.float32), name='g_b_out')
                x_hat = tf.nn.tanh(tf.matmul(hidden2, w_out) + b_out, name='g_x_hat')
                g = tf.reshape(x_hat, [batch_size] + list(FLAGS.image_shape))

                # [end] Based on https://github.com/AshishBora/mnist-vae.
            return g

    @property
    def model_dir(self):
        """Creates a model directory."""

        if not hasattr(self, 'model_dir_str'):  # If model_dir_str has not been set already.
            if os.path.basename(FLAGS.cfg_path) == 'cfg.yml' and not FLAGS.is_train:
                self.model_dir_str = os.path.dirname(FLAGS.cfg_path).split('/')[-1]
            else:
                generic_name = "{}_{}_{}_{}_{}_a0"

                if FLAGS.initialize_from is not None:
                    init_path = FLAGS.initialize_from
                    init_model_name = ''
                    if os.path.isdir(init_path):
                        init_model_name = init_path.split('/')[-1]
                    elif os.path.exists(init_path) and '.yml' in init_path:
                        init_model_name = init_path.replace('.yml', '').split('/')[-1]

                    self.model_postfix += '_INIT_{}'.format(init_model_name)

                generic_name = generic_name + self.model_postfix
                self.model_dir_str = generic_name.format(self.dataset_name, self.batch_size, self.output_height,
                                                         self.output_width, FLAGS.exp_name, FLAGS.a_ind)

        return self.model_dir_str

    def cs_grad_estimator(self, test_batch_size=None):
        """Creates the estimator part for optimizing for z instead of sampling it.
        Related to https://github.com/AshishBora/csgm/blob/master/src/mnist_estimators.py.

        Args:
            test_batch_size: (optional) The test batch size for extracting zs.
        """

        # Get a session.
        sess = self.sess
        # Initialize variables.
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # Set CS example batch size.
        if test_batch_size is not None:
            self.cs_bsize = test_batch_size
        else:
            if not self.cs_m_and_orig:
                self.cs_bsize = FLAGS.cs_batch_size
            else:
                self.cs_bsize = FLAGS.cs_batch_size + FLAGS.batch_size  # When original images are used.

        # Create the generator.
        if self.contrastive_learning:
            self.cs_input_labels = tf.placeholder(tf.int32, shape=(self.cs_bsize), name='cs_input_labels')

        # The z latent variable.
        self.z_batch = tf.Variable(tf.random_normal([self.cs_bsize * FLAGS.cs_num_random_restarts, self.z_dim]),
                                   name='z_batch')

        # Regularizer.
        self.zp_loss_batch = tf.reduce_mean(self.z_batch ** 2, 1)

        # Placeholder for initializing z.
        self.z_init_pl = tf.placeholder(tf.float32, [self.cs_bsize * FLAGS.cs_num_random_restarts, self.z_dim])
        self.z_init_op = tf.assign(self.z_batch, self.z_init_pl)

        cs_batch_size = self.cs_bsize * FLAGS.cs_num_random_restarts
        self.cs_batch_size = cs_batch_size

        # CS loss (The same function is used for encoder input).
        self.cs_loss(self.z_batch, cs_batch_size=cs_batch_size)

        # Set up gradient descent optimizer.
        global_step = tf.Variable(0, trainable=False)
        self.cs_learning_rate = get_cs_learning_rate(global_step)
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            opt = tf.train.AdamOptimizer(self.cs_learning_rate)
            self.cs_update_op = opt.minimize(self.cs_total_loss, var_list=[self.z_batch], global_step=global_step,
                                             name='update_op')

        # Get optimizer variables to re-initialize before each estimator call.
        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                print '[WARNING] UNINIT {}'.format(var.op.name)
                uninitialized_vars.append(var)
        self.cs_initialize_op = tf.variables_initializer(uninitialized_vars)

    def estimator(self, target_xs, cs_labels=None, z_init_val=None):
        """Creates the estimator function for reconstructing the inpts.
        Modified version of https://github.com/AshishBora/csgm/blob/master/src/mnist_estimators.py

        Args:
            target_xs: Tensor of original samples that will turn into measurements y = self.A * target_xs
            cs_labels: Labels of target_xs for the contrastive loss regularizer.
            z_init_val: Initial value of z.

        Returns:
            A functional that returns reconstructions of the input images.
        """

        # Initialize z.
        if z_init_val is not None:
            self.sess.run([self.z_init_op], feed_dict={self.z_init_pl: z_init_val})

        # Prepare feed_dict.
        feed_dict = {self.xs_target: target_xs}
        if cs_labels is not None:
            feed_dict = {self.xs_target: target_xs, self.cs_input_labels: cs_labels.astype(np.int32)}

        self.sess.run(self.cs_initialize_op, feed_dict=feed_dict)

        for j in range(FLAGS.cs_max_update_iter):
            _, lr_val, total_loss_val, all_x_hat_batch_val, \
            all_total_loss_batch_val, all_z_hat, y_batch_val, \
            m_loss1_val, m_loss2_val, zp_loss_val, cont_loss_val = self.sess.run(
                [self.cs_update_op, self.cs_learning_rate, self.cs_total_loss, self.x_hat_batch,
                 self.cs_total_loss_batch, self.z_batch, self.y_batch,
                 self.m_loss1, self.m_loss2, self.zp_loss, self.cont_loss],
                feed_dict=feed_dict)

        logging_format = 'cs: rr{} iter {} lr {} total_loss {} m_loss1 {} m_loss2 {} zp_loss {} cont_loss {}'

        print logging_format.format(FLAGS.cs_num_random_restarts, j, lr_val, total_loss_val, m_loss1_val, m_loss2_val,
                                    zp_loss_val, cont_loss_val)

        # Write the summaries for disc loss.
        cs_sum_strs = self.sess.run(self.cs_sum, feed_dict=feed_dict)
        self.writer.add_summary(cs_sum_strs, self.counter)

        # Get the best reconstructions from random restarts.
        ret_zs = np.zeros([self.cs_bsize, self.z_dim])
        ret_x_hats = np.zeros([self.cs_bsize] + list(all_x_hat_batch_val.shape[1:]))
        ret_ys = np.zeros([self.cs_bsize, FLAGS.cs_num_measurements])
        best_inds = np.zeros([self.cs_bsize], dtype=np.int32)
        labels = np.zeros([self.cs_bsize], dtype=np.int32)

        item_inds = []
        for i in range(self.cs_bsize):
            cur_item_inds = []
            for j in range(FLAGS.cs_num_random_restarts):
                inds = [i + j * self.cs_bsize]
                cur_item_inds.append(inds)
            cur_item_inds = np.concatenate(cur_item_inds).flatten()
            item_inds.append(cur_item_inds)

        for i in range(self.cs_bsize):
            if item_inds[i].shape[0] == 1:
                best_inds[i] = i
            else:
                best_inds[i] = item_inds[i][np.argmin(all_total_loss_batch_val[item_inds[i]])].astype(int)
            ret_zs[i, :] = all_z_hat[best_inds[i]]
            ret_ys[i, :] = y_batch_val[i]
            ret_x_hats[i, :] = all_x_hat_batch_val[best_inds[i]]

        return [ret_x_hats, ret_zs]

    def get_x_hats(self, all_zs):
        """Runs the generator on all_zs.

        Args:
             all_zs: The input zs.

        Returns:
            The np.ndarray of images.
        """

        batch_size = int(self.G.get_shape()[0])
        x_hats = []
        G = self.generator(self.z, train=False, reuse=True, batch_size=batch_size)
        num_iters = int(np.ceil(all_zs.shape[0] * 1.0 / batch_size))

        for bidx in range(num_iters):
            batch_zs = all_zs[batch_size * bidx:(bidx + 1) * batch_size]
            s = batch_zs.shape[0]
            rem = batch_size - batch_zs.shape[0]
            if rem > 0:
                temp = np.zeros([rem, batch_zs.shape[1]], dtype=np.float32)
                batch_zs = np.concatenate([batch_zs, temp])
            [x_hat_batch_val] = self.sess.run([G], feed_dict={self.z: batch_zs})

            if rem > 0:
                x_hat_batch_val = x_hat_batch_val[:s]
            x_hats.append(x_hat_batch_val)

        return np.concatenate(x_hats)

    def guarantee_initialized_variables(self):
        """Initializes the uninitialized variables."""

        global_vars = tf.global_variables()
        is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

        for x in ['[#] Initialized: ' + str(i.name) for i in
                  not_initialized_vars]:
            print(x)

        if len(not_initialized_vars):
            self.sess.run(tf.variables_initializer(not_initialized_vars))
            return True
        else:
            return False

    def train(self):
        """Trains CSGAN/DCGAN."""

        # Prepares the dataset.
        ds = load_ds()
        data_X, data_Y, _ = ds.load(FLAGS.train_split)
        # DCGAN training with a specified number of training data (not the entire dataset).
        if not self.cs_learning and hasattr(FLAGS, 'orig_size'):
            data_X = data_X[:FLAGS.orig_size]
            try:
                data_Y = data_Y[:FLAGS.orig_size]
            except:
                print('[WARNING] NO LABELS FOR {}'.format(FLAGS.dataset))

        # Discriminator and generator optimizers.
        d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        self.guarantee_initialized_variables()

        # Summaries.
        g_list = [self.d_loss_fake_sum, self.g_loss_sum]
        if self.cs_learning:
            self.CS_X_hat_sum = image_summary("X_HAT", self.x_hat_batch)
            g_list.append(self.CS_X_hat_sum)

        self.g_sum = merge_summary(g_list)
        self.d_sum = merge_summary([self.d_loss_real_sum, self.d_loss_sum])

        # Log dir within checkpoint dir.
        log_dir = os.path.join(self.checkpoint_dir, self.model_dir, 'logs')

        ensure_dir(log_dir)

        # Creating the summary writer.
        if FLAGS.tensorboard_log:
            self.writer = SummaryWriter(log_dir, self.sess.graph)
        else:
            self.writer = DummyWriter()  # Does nothing except implementing the tensorboard's logging interface.

        start_time = time.time()  # Time keeping.

        # The total batch size.
        all_batch_size = FLAGS.batch_size + FLAGS.cs_batch_size
        if FLAGS.initialize_from is not None:
            print('[#] Should be already pre-initialized.')
            checkpoint_counter = self.counter
            could_load = True
        else:
            could_load, checkpoint_counter = self.load(FLAGS.checkpoint_dir)
        if FLAGS.retrain:
            if could_load:
                print('[!] Found model will be wiped.')
            could_load = False
            checkpoint_counter = 0

        if self.cs_m_and_orig:
            batch_idxs = (len(data_X) - FLAGS.orig_size) // all_batch_size
            orig_idx = 1
        else:
            batch_idxs = len(data_X) // all_batch_size

        # Loading model.
        cur_epoch = 0
        if could_load:
            self.counter = checkpoint_counter
            cur_epoch = ((self.counter + 1) // batch_idxs)
            print(" [*] Load SUCCESS")
        else:
            self.counter = 0
            print(" [!] Load failed...")

        # Iteration calcucations.
        epochs = FLAGS.epoch

        self.cur_epoch = cur_epoch
        # Main training loop.
        for epoch in xrange(cur_epoch, epochs):
            for idx in xrange(self.counter % batch_idxs, batch_idxs):
                if self.cs_m_and_orig:  # Both compressed and non-compressed data.
                    orig_idx = (orig_idx + 1) % (
                            FLAGS.orig_size // FLAGS.batch_size)
                    cs_batch_images = data_X[
                                      FLAGS.orig_size + idx * FLAGS.cs_batch_size:
                                      FLAGS.orig_size + (
                                              idx + 1) * FLAGS.cs_batch_size]
                    cs_batch_images = ds.transform(cs_batch_images)

                    orig_batch_images = data_X[orig_idx * FLAGS.batch_size:
                                               (orig_idx + 1) * FLAGS.batch_size]
                    orig_batch_images = ds.transform(orig_batch_images)
                    all_batch_images = np.concatenate(
                        [cs_batch_images, orig_batch_images])
                else:  # Only one type of data (compressed only or non-compressed only)
                    all_batch_images = data_X[idx * all_batch_size:(
                                                                           idx + 1) * all_batch_size]
                    all_batch_images = ds.transform(all_batch_images)

                    if self.cs_learning:
                        cs_batch_images = all_batch_images[FLAGS.batch_size:]

                if self.contrastive_learning:
                    assert not self.cs_m_and_orig
                    all_batch_labels = data_Y[idx * FLAGS.cs_batch_size:(
                                                                                idx + 1) * FLAGS.cs_batch_size]

                batch_z = np.random.uniform(-1, 1, [FLAGS.batch_size,
                                                    self.z_dim]).astype(
                    np.float32)
                # CSGAN.
                if self.cs_learning:
                    ## Find the zs with contrastive loss. Then append ranodm
                    ## zs to them. Note: this is not included in the paper
                    ## and the related configurations have BATCH_SIZE = 0.
                    if self.contrastive_learning:
                        assert not self.cs_m_and_orig
                        _, z_hats = self.estimator(cs_batch_images,
                                                   cs_labels=all_batch_labels)
                        all_z = np.concatenate((batch_z, z_hats))
                    else:
                        if self.cs_m_and_orig:
                            _, z_hats = self.estimator(all_batch_images)
                            all_z = z_hats
                        else:
                            _, z_hats = self.estimator(cs_batch_images)
                            all_z = np.concatenate((batch_z, z_hats))
                # DCGAN.
                else:
                    all_z = batch_z

                feed_dict = {}
                feed_dict[self.z] = all_z

                # DCGAN or CSGAN with just compressed measurements (inputs are always the compressed inputs).
                if not self.cs_learning or not FLAGS.just_cs:
                    feed_dict[self.inputs] = all_batch_images

                # CSGAN.
                if self.cs_learning:
                    # CSGAN + compressed measurements and non-compressed inputs.
                    if self.cs_m_and_orig:
                        feed_dict[self.orig_inputs] = orig_batch_images
                        feed_dict[self.xs_target] = all_batch_images
                    # CSGAN + only one type of inputs.
                    else:
                        # xs_target are the sensed inputs.
                        feed_dict[self.xs_target] = cs_batch_images

                    if self.contrastive_learning:
                        assert not self.cs_m_and_orig
                        feed_dict[self.cs_input_labels] = all_batch_labels

                # Run the optimizer for the discriminator.
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict=feed_dict)
                self.writer.add_summary(summary_str, self.counter)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict=feed_dict)
                self.writer.add_summary(summary_str, self.counter)

                # Run the optimizer for the generator.
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict=feed_dict)
                self.writer.add_summary(summary_str, self.counter)
                errD_fake = self.d_loss_fake.eval(feed_dict)
                errD_real = self.d_loss_real.eval(feed_dict)
                errG = self.g_loss.eval(feed_dict)

                self.counter += 1

                print(
                        "Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                        % (epoch, FLAGS.epoch, idx, batch_idxs,
                           time.time() - start_time, errD_fake + errD_real, errG))

                # Save model.
                if self.counter % max(1, int(batch_idxs // 2)) == 0:
                    self.save(FLAGS.checkpoint_dir, self.counter)

        # Save at the end.
        self.save(FLAGS.checkpoint_dir, self.counter)

    def save(self, checkpoint_dir, step):
        """Saves the current weights.

        Args:
            checkpoint_dir: The path to the checkpoint directory.
            step: Current training step.
        """

        model_name = "CSGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load_from_path(self, checkpoint_dir):
        """Loads the weights from a checkpoint file.

        Args:
            checkpoint_dir: Checkpoint directory.

        Returns:
            True if load is sucessful, False otherwise.
        """

        vars = self.save_var_names
        saver = tf.train.Saver(vars)

        def load_aux(ckpt_path):
            """Helper function to not repeat the same code in the following lines."""

            ckpt_name = os.path.basename(ckpt_path)
            saver.restore(self.sess, ckpt_path)
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            self.counter = counter
            print(" [*] Loaded {}".format(ckpt_name))
            return True, counter

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        try:
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                return load_aux(os.path.join(checkpoint_dir, ckpt_name))
            else:
                print(
                    " [!] Failed to find a checkpoint within directory {}".format(
                        FLAGS.ckpt_path))
                return False, 0
        except:
            print(" [!] Failed to find a checkpoint, Exception!")
            return False, 0

    def load(self, checkpoint_dir=None):
        """Loads a saved model.

        Args:
            checkpoint_dir: Root of all the checkpoints

        Returns:
            True if load is successful.
            The counter showing the iteration.
        """

        if checkpoint_dir is None:
            checkpoint_dir = FLAGS.checkpoint_dir

        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        return self.load_from_path(checkpoint_dir)

    @property
    def training_dataset_size(self):
        """Gets size of the training data."""

        if not self.cs_learning and hasattr(FLAGS, 'orig_size'):
            return FLAGS.orig_size

        traindata_size_dir = os.path.join(self.cache_path, 'ds_sizes')
        ensure_dir(traindata_size_dir)
        if not hasattr(FLAGS, 'train_split'):
            setattr(FLAGS, 'train_split', 'train')

        size_cache_file = os.path.join(traindata_size_dir, '{}_{}'.format(FLAGS.dataset.lower(), FLAGS.train_split))

        if os.path.exists(size_cache_file):
            with open(size_cache_file) as f:
                ds_size = int(f.readline().strip())
        else:
            ds = load_ds()  # Loads the dataset.
            [data_X, _, _] = ds.load()
            ds_size = len(data_X)
            with open(size_cache_file, 'w') as f:
                f.write(str(ds_size))

        return ds_size

    @property
    def output_dir(self):
        """Returns the output directory for this object."""
        return os.path.join(self.checkpoint_dir, self.model_dir)

    @property
    def training_finished(self):
        """A property to check if the network is trained on all epochs.

        Returns:
            True if it's trained, False otherwise.
        """
        finished_flag_file = os.path.join(self.output_dir, 'finished.info')

        could_load, checkpoint_counter = self.load(FLAGS.checkpoint_dir)
        self.cur_epoch = 0
        self.counter = 0
        if hasattr(self, 'counter') and could_load:
            all_batch_size = FLAGS.batch_size + FLAGS.cs_batch_size

            if could_load:
                self.counter = checkpoint_counter
                self.cur_epoch = ((all_batch_size * (self.counter + 1)) // self.training_dataset_size)
                print("[*] Load success.")

            if self.cur_epoch < FLAGS.epoch - 1:
                return False
            else:
                with open(finished_flag_file, 'w') as f:
                    f.write('1')
                return True
        else:
            return False
