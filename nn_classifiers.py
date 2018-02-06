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

"""Contains the Neural Network Classifier class."""

import os
import math
from ops import *
import numpy as np
import glob

from utils.csgan_utils import DummyWriter


def conv_out_size_same(size, stride):
    """This function calculates the output size of a convolutional layer.

    Args:
        size: Input size to the convolutional layer.
        stride: Stride of the convolution filter.

    Returns:
        Convolutional layer output size.
    """
    return int(math.ceil(float(size) / float(stride)))


class NNClassifier(object):
    """Neural Network Classifier class."""

    def __init__(self, network_name, input_dim, num_classes, initial_lr, batch_size,
                 num_epochs, checkpoint_dir, optimizer_type='decay_sgd', num_hidden_units=None, num_hidden_layers=None,
                 use_batch_norm=False):
        """Neural Network Classifier constructor.

        Args:
            network_name: Architecture name. Currently supports [mlp|lenet|lenet_map].
            input_dim: Dimension of the input layer.
            num_classes: Number of classes (also dimension of the output layer).
            initial_lr: Initial learning rate.
            batch_size: Batch size.
            num_epochs: Number of training epochs.
            checkpoint_dir: Checkpoint directory (to save TensorFlow checkpoints in).
            optimizer_type: Optimizer type. Currently supports [decay_sgd|fixed_lr_sgd|adam] (Decay SGD, Fixed learning
            rate SGD, and Adam optimizers, respectively).
            num_hidden_units: Array of number of hidden units (for each hidden layer). Only need for network_name = 'mlp'
            (multi-layer perceptron).
            num_hidden_layers: Number of hidden layers. Only need for network_name = 'mlp' (multi-layer perceptron).
            use_batch_norm: Boolean. Whether or not to use batch normalization when training.

        Raises:
            RuntimeError: If the input parameters are invalid.
        """

        # Initialize attributes.
        self.network_name = network_name
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_hidden_units = num_hidden_units
        self.num_hidden_layers = num_hidden_layers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.initial_lr = initial_lr
        self.use_batch_norm = use_batch_norm
        self.optimizer_type = optimizer_type
        if network_name == 'mlp' and len(num_hidden_units) < num_hidden_layers:
            raise RuntimeError('[!] Invalid parameters.')
        self.build_model()

        # Add all classifier variables to saver.
        tf_vars = tf.global_variables()
        self.cl_vars = [var for var in tf_vars if 'cl_' in var.name]
        self.saver = tf.train.Saver(var_list=self.cl_vars, max_to_keep=None)
        self.summary_writer = DummyWriter()

    def cross_entropy(self, logits, labels):
        """Cross entropy loss.

        Args:
            logits: Logits (output of neural network).
            labels: Ground-truth labels.

        Returns:
            cross_entropy: Cross-entropy loss.

        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
        return tf.reduce_mean(cross_entropy, name='xentropy_mean')

    def build_model(self):
        """Builds the network architecture and defines the input and output placeholders.

        Raises:
            ValueError: If the specified network name is not supported. Currently supports [mlp|lenet|lenet_map].
        """

        # Input placeholder.
        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.input_dim], name='input_features')
        # Ground-truth labels.
        self.labels = tf.placeholder(tf.int32, None, name='ground_truth_labels')
        # Keep probability placeholder if dropout is used.
        self.keep_prob = tf.placeholder(tf.float32)

        # Define train and test outputs.
        # Multi-layer perception.
        if self.network_name == 'mlp':
            self.output = self.mlp()
            self.test_output = self.mlp(reuse=True)
        # LeNet.
        elif self.network_name == 'lenet':
            self.output = self.lenet(True, map_flag=False)
            self.test_output = self.lenet(False, map_flag=False, reuse=True)
        # LeNet + Map. This is the LeNet architecture with an additional layer which maps the input to the input size
        # expected by the LeNet network (28 * 28 = 784)
        elif self.network_name == 'lenet_map':
            self.output = self.lenet(True, map_flag=True)
            self.test_output = self.lenet(False, map_flag=True, reuse=True)
        else:
            raise ValueError('[!] Network name {} is not supported.'.format(self.network_name))

        # Define loss and global step.
        self.loss = self.cross_entropy(self.output, self.labels)
        self.loss_sum = scalar_summary("loss", self.loss)
        self.global_step = tf.Variable(tf.constant(0), trainable=False, name='cl_global_step')

    def mlp(self, reuse=False):
        """Defines the multi-layer perceptron input-output relationship.

        Args:
            reuse: When set to True, re-uses TensorFlow variables.

        Returns:
            Output of MLP.
        """

        with tf.variable_scope('mlp') as scope:
            if reuse:
                scope.reuse_variables()

            outputs = [self.inputs]
            # For every hidden layer, pass preceding layer into a linear layer with a leaky ReLU activation function.
            for i in range(self.num_hidden_layers):
                outputs.append(lrelu(linear(outputs[i], self.num_hidden_units[i], 'hidden{}'.format(i))))
            outputs.append(lrelu(linear(outputs[-1], self.num_classes, 'output_layer')))
            return outputs[self.num_hidden_layers + 1]

    def lenet_bn(self, inputs, train=True, reuse=False, map_flag=False):
        """Defines the LeNet with batch-normalization input-output relationship.

        Args:
            inputs: LeNet input.
            train: Train (True) or test (False) mode.
            reuse: Whether or not to reuse TensorFlow variables (False the first time the network is called).
            map_flag: If True, defines an additional layer which maps the input to the input size expected by the LeNet
            network (28 * 28 = 784) (lenet_map).

        Returns:
            Output of LeNet with batch normalization.
        """
        if not reuse:
            self.bn1 = batch_norm(name='cl_bn1')
            self.bn2 = batch_norm(name='cl_bn2')

        with tf.variable_scope('cl_lenet_bn') as scope:
            if reuse:
                scope.reuse_variables()

            # When map_flag is True, a linear layer is defined between the input and the LeNet network.
            if map_flag:
                inputs = linear(inputs, 28 * 28)

            # Define the LeNet architecture with batch normalization and dropout.
            h0_reshape = tf.reshape(inputs, [-1, 28, 28, 1])
            h1 = lrelu(self.bn1(conv2d(h0_reshape, 20, k_h=5, k_w=5, d_h=1, d_w=1, name='cl_conv1'), train=train))
            h1_pool = tf.contrib.slim.max_pool2d(h1, [2, 2], 2)
            h2 = lrelu(self.bn2(conv2d(h1_pool, 50, k_h=5, k_w=5, d_h=1, d_w=1, name='cl_conv2'), train=train))
            h2_pool = tf.contrib.slim.max_pool2d(h2, [2, 2], 2)
            h2_reshape = tf.reshape(h2_pool, [-1, 7 * 7 * 50])
            h3 = linear(h2_reshape, 500, scope='cl_fc1')
            h3_drop = tf.nn.dropout(h3, self.keep_prob)
            h4 = linear(h3_drop, self.num_classes, scope='cl_fc2')
        return h4

    def lenet_nobn(self, inputs, reuse=False, map_flag=False):
        """
        Defines the LeNet with no batch-normalization input-output relationship.

        Args:
            inputs: LeNet input.
            reuse: Whether or not to reuse TensorFlow variables (False the first time the network is called).
            map_flag: If True, defines an additional layer which maps the input to the input size expected by the LeNet
            network (28 * 28 = 784) (lenet_map).

        Returns:
            Output of LeNet without batch normalization.
        """
        with tf.variable_scope('lenet') as scope:
            if reuse:
                scope.reuse_variables()

            # When map_flag is True, a linear layer is defined between the input and the LeNet network.
            if map_flag:
                inputs = linear(inputs, 28 * 28)

            # Define the LeNet architecture with batch normalization and no dropout.
            h0_reshape = tf.reshape(inputs, [-1, 28, 28, 1])
            h1 = lrelu(conv2d(h0_reshape, 20, k_h=5, k_w=5, d_h=1, d_w=1, name='conv1'))
            h1_pool = tf.contrib.slim.max_pool2d(h1, [2, 2], 2)
            h2 = lrelu(conv2d(h1_pool, 50, k_h=5, k_w=5, d_h=1, d_w=1, name='conv2'))
            h2_pool = tf.contrib.slim.max_pool2d(h2, [2, 2], 2)
            h2_reshape = tf.reshape(h2_pool, [-1, 7 * 7 * 50])
            h3 = linear(h2_reshape, 500, scope='fc1')
            h4 = linear(h3, self.num_classes, scope='fc2')
        return h4

    def lenet(self, train, map_flag=False, reuse=False):
        """Defines the LeNet input-output relationship.

        Args:
            train: Train (True) or test (False) mode.
            map_flag: If True, defines an additional layer which maps the input to the input size expected by the LeNet
            network (28 * 28 = 784) (lenet_map).
            reuse: Whether or not to reuse TensorFlow variables.

        Returns:
            Output of LeNet.
        """
        # Call appropriate method depending on self.use_batch_norm.
        if self.use_batch_norm:
            return self.lenet_bn(self.inputs, train=train, reuse=reuse, map_flag=map_flag)
        else:
            return self.lenet_nobn(self.inputs, reuse=reuse, map_flag=map_flag)

    def save(self):
        """Saves TensorFlow model intro checkpoint directory."""

        # Define model name.
        model_name = self.network_name.upper() + '.model'
        # Create checkpoint directory if it does not already exist.
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # Save model.
        self.saver.save(self.session, os.path.join(self.checkpoint_dir, model_name), global_step=self.global_step)

    def load(self, ckpt_name=None):
        """Loads TensorFlow model from checkpoint.

        Args:
            ckpt_name: Path to TensorFlow checkpoint.

        Returns:
            True: If load succeeded.
            False: If unable to load a model.
        """

        # If not checkpoint name was specified, load latest model from checkpoint directory.
        if ckpt_name is None:
            print("[*] Reading checkpoints.")
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.saver.restore(self.session, os.path.join(self.checkpoint_dir, ckpt_name))
                print("[*] Success loading {}.".format(ckpt_name))
                return True
            else:
                print("[*] Failed to find a checkpoint at {}.".format(self.checkpoint_dir))
                return False

        # If a checkpoint name was specified, restore from it.
        else:
            self.saver.restore(self.session, ckpt_name)
            print("[*] Success loading {}.".format(os.path.basename(ckpt_name)))
            return True

    def get_learning_rate(self):
        """Returns learning rate at the current global step.

        Returns:
            Learning rate.
        """
        # Adam and the fixed learning rate SGD optimizers have a fixed learning rate. Return a constant equal to the
        # initial learning rate.
        if self.optimizer_type in {'adam', 'fixed_lr_sgd'}:
            return tf.constant(self.initial_lr)
        # For decay SGD, use exponential decay of the learning rate.
        else:
            return tf.train.exponential_decay(self.initial_lr, self.global_step, self.steps_per_epoch * 40, 0.1,
                                              staircase=True)

    def fit(self, features, labels, retrain=False, session=None):
        """Fits the neural network weights, given the input features and labels.

        Args:
            features: Train features.
            labels: Ground-truth training labels.
            retrain: When True, re-train from scratch (rather than resume training).
            session: TensorFlow session.

        Raises:
            ValueError: If the optimizer type in self.optimizer_type is not supported. Currently supports
            [decay_sgd|fixed_lr_sgd|adam].
        """

        self.session = session
        # Number of steps per epoch. One epoch is an entire pass through the data.
        self.steps_per_epoch = features.shape[0] // self.batch_size
        self.learning_rate = self.get_learning_rate()
        # Define optimizers.
        if self.optimizer_type in {'decay_sgd', 'fixed_lr_sgd'}:
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss=self.loss,
                                                                                       global_step=self.global_step)
        elif self.optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss=self.loss,
                                                                            global_step=self.global_step)
        else:
            raise ValueError('[!] Optimizer type {} is not supported.'.format(self.network_name))

        # Initialize uninitialized variables.
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # Create log directory if not present.
        log_dir = os.path.join(self.checkpoint_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.writer = DummyWriter()
        cur_epoch = 0

        # When retrain is False, start by loading model.
        if not retrain:
            try:
                could_load = self.load()
                if could_load:
                    # If retraining, initialize the current epoch based on the global step.
                    cur_epoch = (self.batch_size * self.global_step.eval()) // features.shape[0]
                    print("[*] Load success.")
                else:
                    print("[!] Load failed.")
            except:
                print("[!] Load failed.")
        else:
            print("[*] Re-training.")

        batch_idxs = np.floor(features.shape[0] * 1.0 / self.batch_size).astype(int)
        for epoch in xrange(cur_epoch, self.num_epochs):
            for idx in xrange(self.global_step.eval() % batch_idxs, batch_idxs):
                # At every batch of every epoch, get the corresponding features and labels, and run the optimizer.
                batch_features = features[idx * self.batch_size:(idx + 1) * self.batch_size, :]
                batch_features = batch_features.reshape([self.batch_size, -1])
                batch_labels = labels[idx * self.batch_size:(idx + 1) * self.batch_size]
                _, summary_str, lr, loss = self.session.run([optimizer, self.loss_sum, self.learning_rate, self.loss],
                                                         feed_dict={self.inputs: batch_features,
                                                                    self.labels: batch_labels,
                                                                    self.keep_prob: 0.5})
            print('Epoch: {}, Learning rate: {}, loss: {}.'.format(epoch, lr, loss))
            # Save model after every epoch.
            self.save()

    def predict(self, features, model_name=None, session=None):
        """Predicts labels given test feature vectors.

            Args:
                features: Test feature vectors.
                model_name: Optional, model name to load and use to predict.
                session: TensorFlow session.

            Returns:
                Array of predicted labels.

            Raises:
                RuntimeError: If unable to load the model specified in model_name.
        """
        self.session = session
        # Initialize uninitialized variables.
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # If a model name was specified, try to load it.
        if model_name is not None:
            could_load = self.load(model_name)
            log_dir = os.path.join(self.checkpoint_dir, 'logs')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            if could_load:
                print("[*] Load success.")
            else:
                raise RuntimeError('[!] Unable to load model {}.'.format(model_name))

        cur_idx = 0
        labels = []
        batch_idxs = np.ceil(features.shape[0] * 1.0 / self.batch_size).astype(int)
        # At every batch of every epoch, get the corresponding features and labels, and run the network to obtain
        # predictions.
        for idx in xrange(cur_idx, batch_idxs):
            batch_features = features[idx * self.batch_size:(idx + 1) * self.batch_size, :]
            curr_batch_size = batch_features.shape[0]
            batch_features = batch_features.reshape([curr_batch_size, -1])
            # Last batch may not have correct size, append zeros and discard output.
            if idx == batch_idxs - 1:
                rem = self.batch_size - curr_batch_size
                temp = np.zeros([rem] + list(batch_features.shape[1:]))
                batch_features = np.concatenate([batch_features, temp])
            output_vals = \
                self.session.run([self.test_output], feed_dict={self.inputs: batch_features, self.keep_prob: 1.0})[0]
            if idx == batch_idxs - 1:
                output_vals = output_vals[:curr_batch_size]
            # Append predicted labels.
            labels.append(np.argmax(output_vals, axis=1))

        return np.concatenate(labels)

    def validate(self, features, labels, session=None):
        """Validates different checkpoints by testing them on the validation split and retaining the one with the top
        accuracy.

        Args:
            features: Validation feature vectors.
            labels: Validation ground-truth labels.
            session: TensorFlow session.

        Returns:
            best_accuracy: Accuracy of the chosen model (with the highest accuracy on the validation set).
            best_model_name: Name of the chosen model.
            best_model_index: Index of the chosen model.
        """

        # Initialize best model info.
        best_model_index = -1
        best_model_name = ''
        best_accuracy = 0
        ind = 0

        # Loop over all models and get their accuracy on the validation features.
        for model_file in glob.glob(os.path.join(self.checkpoint_dir, self.network_name.upper() + '.model*.index')):
            model_name = model_file[:model_file.find('.index')]
            predicted_labels = self.predict(features, model_name, session=session)
            num_correct = np.sum(np.equal(predicted_labels, labels))
            accuracy = num_correct / (1.0 * len(labels))
            print('[*] Tested {} on validation set, accuracy: {}.'.format(os.path.basename(model_name), accuracy))

            # Keep model with the highest accuracy.
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
                best_model_index = ind

        return best_accuracy, best_model_name, best_model_index
