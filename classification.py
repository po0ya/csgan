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

"""Contains the Classifier class."""

from __future__ import division

import cPickle
import os
import numpy as np
import re
from sklearn import linear_model, svm
from metric_learn import LMNN
from sklearn import neighbors

from datasets.celeba import CelebA
from datasets.fmnist import FMnist
from datasets.mnist import Mnist
import utils.cls_utils as cls
from models import nn_classifiers as nn


class Classifier(object):
    """Classifier class."""

    def __init__(self, cfg, feature_file=None, test_split='test'):
        """Classifier Constructor. See build_classifier method for more details.

        Args:
            cfg: Path to configuration file.
            feature_file: Path to feature file.
            test_split: Split to test on.

        Raises:
            RuntimeError: If classifier_type is not specified in the config file.
        """

        # Creates a dictionary of params from the config file.
        self.classifier_params = cls.get_cls_param_dict(cfg)

        # Get params from optional args.
        self.classifier_params['feature_file'] = feature_file
        self.classifier_params['feature_dir'] = os.path.dirname(feature_file)
        self.classifier_params['test_split'] = test_split

        # If classifier type was not set, raise an exception.
        if 'classifier_type' not in self.classifier_params:
            raise RuntimeError('[!] No specified classifier type.')

        self.classifier_type = self.classifier_params['classifier_type']
        self.estimator = None  # Actual classifier.
        self.helper_estimator = None  # Only used for metric learning, the helper estimator will learn the metric.
        self.parse_feature_file()
        self.build_classifier()  # Classifier initialization given the params.

    def parse_feature_file(self):
        """Parses feature filename for various parameters.

        Raises:
            RuntimeError: If the feature file is invalid (does not belong to reconstructed images, measurements, or
            latent space variables).
        """

        feature_file = self.classifier_params['feature_file']

        # Checks if feature file is based on reconstructed images (x_hats), measurements (y), or the obtained latent
        # space variable (z_hats).
        if feature_file.find("x_hats") > -1:
            self.classifier_params['input_feature'] = 'x_hats'
        elif feature_file.find("measurements") > -1:
            self.classifier_params['input_feature'] = 'measurements'
        elif feature_file.find("z_hats") > -1:
            self.classifier_params['input_feature'] = 'z_hats'
        else:
            raise RuntimeError('[!] Invalid feature file.')

        # Get different parameters of the experiment.
        self.classifier_params['learning_rate'] = re.search('lr(([0-9]|\.)+)', feature_file).group(1)
        self.classifier_params['random_restarts'] = re.search('rr([0-9]+)', feature_file).group(1)
        self.classifier_params['num_measurements'] = re.search('m([0-9]+)', feature_file).group(1)
        self.classifier_params['counter'] = re.search('c([0-9]+)', feature_file).group(1)
        self.classifier_params['a_index'] = re.search('a([0-9]+)', feature_file).group(1)

    def build_classifier(self):
        """Initializes classifier based on self.classifier_params.

        Raises:
            ValueError: If self.classifier is not supported (currently supports [svm|linear-svm|lmnn|logistic|knn|nn]).
        """

        # Different classifier types are treated differently.

        # Kernel SVM.
        if self.classifier_type == 'svm':
            # Default params.
            params = {'c_penalty': 1.0,  # Penalty parameter of the error term.
                      'kernel': 'rbf',  # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
                      'degree': 3,  # Degree of polynomial for 'poly' kernel.
                      'gamma': 'auto',  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
                      'coef0': 0.0,  # Independent term in kernel for 'poly' and 'sigmoid'.
                      'shrinking': True,  # Whether to use the shrinking heuristic.
                      'probability': False,  # Whether to enable probability estimates.
                      'tol': 0.001,  # Tolerance for stopping criterion.
                      'cache_size': 200,  # Kernel cache (in MB).
                      'class_weight': None,  # {class_label: weight}.
                      'verbose': False,
                      'random_state': None,  # Seed for pseudo random number generator for shuffling data.
                      'max_iter': -1,  # Hard limit on iterations or -1 for no limit.
                      # Multiclass handling.
                      # 'ovo', 'ovr', or None.
                      # 'ovo': one vs one.
                      # 'ovr' one vs rest.
                      # None is 'ovr'.
                      'multi_class': None,
                      'num_classes': 10}  # Number of classes.

            # Update parameters from dictionary of parameters (based on config file).
            params.update(self.classifier_params)

            # Build the classifier (estimator). Kernel SVM is based on sklearn.
            self.estimator = svm.SVC(C=params['c_penalty'], kernel=params['kernel'], degree=params['degree'],
                                     gamma=params['gamma'], coef0=params['coef0'], shrinking=params['shrinking'],
                                     probability=params['probability'], tol=params['tol'],
                                     cache_size=params['cache_size'], class_weight=params['class_weight'],
                                     verbose=params['verbose'], max_iter=params['max_iter'],
                                     decision_function_shape=params['multi_class'], random_state=params['random_state'])

        # Linear SVM; good for large-scale datasets.
        elif self.classifier_type == 'linear-svm':
            # Default params.
            params = {'penalty': 'l2',  # 'l1' or 'l2'. Norm in the penalization.
                      'loss': 'squared_hinge',  # 'hinge' or 'squared_hinge'. Specifies the loss function.
                      # Use dual or primal optimization problem. Prefer dual=False when n_samples > n_features.
                      'dual': True,
                      'tol': 1e-4,  # Tolerance for stopping criteria.
                      'c_penalty': 1.0,  # Penalty parameter C of the error term.
                      'multi_class': 'ovr',  # 'ovr' (one-vs-rest) or 'crammer_singer' (joint objective in all classes).
                      # Whether or not to calculate the intercept (if false, data is expected to be centered).
                      'fit_intercept': True,
                      'intercept_scaling': 1.0,
                      'class_weight': None,  # {class_label: weight}.
                      'verbose': 0,
                      'random_state': None,  # Seed for random number generator.
                      'max_iter': 1000,  # Maxiumum number of iterations.
                      'num_classes': 10}  # Number of classes.

            # Update parameters from dictionary of parameters (based on config file).
            params.update(self.classifier_params)

            # Build the classifier (estimator). Linear SVM is based on sklearn.
            self.estimator = svm.LinearSVC(penalty=params['penalty'], loss=params['loss'], dual=params['dual'],
                                           tol=params['tol'], C=params['c_penalty'], multi_class=params['multi_class'],
                                           fit_intercept=params['fit_intercept'],
                                           intercept_scaling=params['intercept_scaling'],
                                           class_weight=params['class_weight'], verbose=params['verbose'],
                                           random_state=params['random_state'], max_iter=params['max_iter'])

        # Large Margin nearest neighbor (metric learning + k-nearest neighbor).
        elif self.classifier_type == 'lmnn':
            # Default params.
            # First, metric learning params.
            params = {'num_neighbors': 3,  # Number of neighbors to consider (does not include self-edges).
                      'min_iter': 50,
                      'max_iter': 1000,
                      'learn_rate': 1e-07,
                      'regularization': 0.5,  # Weight of pull and push terms.
                      'tol': 0.001,  # Convergence tolerance.
                      'verbose': False,
                      # Second, k-nn params.
                      # Weights: Callable,  or:
                      # 'uniform': Uniform weights.  All points in each neighborhood are weighted equally.
                      # 'distance': Weigh points by the inverse of their distance.
                      'weights': 'uniform',
                      # Algorithm: {'auto', 'ball_tree', 'kd_tree', 'brute'} 'auto' will attempt to decide the
                      # most appropriate algorithm based on training data.
                      'algorithm': 'auto',
                      'leaf_size': 30,  # Leaf size passed to BallTree or KDTree.
                      'num_jobs': 1,  # The number of parallel jobs to run for neighbors search. -1 -> nb of CPU cores.
                      'num_classes': 10}  # Number of classes.

            # Update parameters from dictionary of parameters (based on config file).
            params.update(self.classifier_params)

            # Build the helper (helper_estimator). Based on the metric_learn package.
            self.helper_estimator = LMNN(k=params['num_neighbors'], min_iter=params['min_iter'],
                                         max_iter=params['max_iter'], learn_rate=params['learn_rate'],
                                         regularization=params['regularization'], convergence_tol=params['tol'],
                                         verbose=params['verbose'])

            # Build the classifier (estimator). Use euclidean distance as a metric. K-NN classifier is based on sklearn.
            self.estimator = neighbors.KNeighborsClassifier(n_neighbors=params['num_neighbors'],
                                                            weights=params['weights'], algorithm=params['algorithm'],
                                                            leaf_size=params['leaf_size'], p=2, metric='minkowski',
                                                            metric_params=None, n_jobs=params['num_jobs'])

        # Logistic regression.
        elif self.classifier_type == 'logistic':
            # Default params.
            params = {'penalty': 'l2',  # 'l1' or 'l2', specify the norm used in the penalization.
                      'dual': False,  # Dual or primal formulation. dual=False is better when n_samples > n_features.
                      'tol': 0.0001,  # Tolerance for stopping criteria.
                      # Inverse of regularization strength (smaller values -> stronger regularization).
                      'c_penalty': 1.0,
                      'fit_intercept': True,  # If a bias should be added to the decision function.
                      'intercept_scaling': 1,
                      'class_weight': None,  # In the form {class_label: weight}.
                      'random_state': None,  # Seed of random number generator for shuffling the data.
                      'solver': 'liblinear',  # 'newton-cg', 'lbfgs', 'liblinear', or 'sag'.
                      'max_iter': 100,  # Maximum number of iterations for the solvers.
                      # Multiclass handling.
                      # 'ovr' one-vs-rest or 'multinomial' If the option chosen is 'ovr', then a binary problem is fit
                      # for each label.
                      # Else the loss minimised is the multinomial loss fit across the entire probability distribution.
                      # Works only for the 'newton-cg', 'sag' and 'lbfgs' solver.
                      'multi_class': 'ovr',
                      'verbose': 0,
                      'warm_start': False,  # Reuse solution of the previous call to fit as initialization.
                      'num_jobs': 1,  # Number of CPU cores during cross-validation. -1 -> all cored are used.
                      'num_classes': 10}  # Number of classes.

            # Update parameters from dictionary of parameters (based on config file).
            params.update(self.classifier_params)

            # Build the classifier (estimator). Logistic regression is based on sklearn.
            self.estimator = linear_model.LogisticRegression(penalty=params['penalty'], dual=params['dual'],
                                                             tol=params['tol'], C=params['c_penalty'],
                                                             fit_intercept=params['fit_intercept'],
                                                             intercept_scaling=params['intercept_scaling'],
                                                             class_weight=params['class_weight'],
                                                             random_state=params['random_state'],
                                                             solver=params['solver'], max_iter=params['max_iter'],
                                                             multi_class=params['multi_class'],
                                                             verbose=params['verbose'], warm_start=params['warm_start'],
                                                             n_jobs=params['num_jobs'])

        # K-Nearest Neighbor classifier (no metric learning).
        elif self.classifier_type == 'knn':
            # Default params.
            params = {'num_neighbors': 3,  # Number of neighbors to use.
                      # Weights: callable,  or:
                      # 'uniform' uniform weights.  All points in each neighborhood are weighted equally.
                      # 'distance' : weigh points by the inverse of their distance.
                      'weights': 'uniform',
                      # Algorithm: {'auto', 'ball_tree', 'kd_tree', 'brute'} 'auto' will attempt to decide the most
                      # appropriate algorithm based on training data
                      'algorithm': 'auto',
                      'leaf_size': 30,  # Leaf size passed to BallTree or KDTree.
                      # Metric: string or DistanceMetric object (default = 'minkowski'), the distance metric to use for
                      # the tree.  The default metric is minkowski, and with p=2 is equivalent to the Euclidean metric.
                      # See the documentation of DistanceMetric.
                      'metric': 'minkowski',
                      'metric_params': None,  # Additional keyword arguments for the metric function.
                      'power': 2,  # Power parameter for the Minkowski metric. p = 1 is l1, p = 2 is l2.
                      'num_jobs': 1,  # The number of parallel jobs to run for neighbors search. -1 -> nb of CPU cores.
                      'num_classes': 10}  # Number of classes.

            # Update parameters from dictionary of parameters (based on config file).
            params.update(self.classifier_params)

            # Build the classifier (estimator). KNN is based on sklearn.
            self.estimator = neighbors.KNeighborsClassifier(n_neighbors=params['num_neighbors'],
                                                            weights=params['weights'], algorithm=params['algorithm'],
                                                            leaf_size=params['leaf_size'], p=params['power'],
                                                            metric=params['metric'],
                                                            metric_params=params['metric_params'],
                                                            n_jobs=params['num_jobs'])

        # Neural network classifier.
        elif self.classifier_type == 'nn':
            # Default params.
            params = {'network_name': 'mlp',  # Name of architecture (should be implemented in NNClassifier.
                      'num_hidden_layers': 3,  # Number of layers (only used if mlp).
                      'num_hidden_units': [200, 200, 10],  # Number of hidden units for each layer (only used if mlp).
                      'num_classes': 10,  # Number of classes.
                      'input_dim': 20,  # Dimension of input layer.
                      'initial_lr': 0.01,  # Initial learning rate.
                      'batch_size': 200,  # Batch size.
                      'num_epochs': 25,  # Number of epochs for training.
                      'optimizer_type': 'decay_sgd',  # Optimizer type.
                      'use_batch_norm': False,  # Whether or not to use batch normalization.
                      # Checkpoint directory: where to save tensorflow checkpoints.
                      'checkpoint_dir': os.path.join(self.get_output_dir(), self.tf_checkpoint_dir())}

            # Update parameters from dictionary of parameters (based on config file).
            params.update(self.classifier_params)

            # Build the classifier (estimator). Neural network classifier is based on the NNClassifier class.
            self.estimator = nn.NNClassifier(network_name=params['network_name'], input_dim=params['input_dim'],
                                             num_hidden_units=params['num_hidden_units'],
                                             num_hidden_layers=params['num_hidden_layers'],
                                             num_classes=params['num_classes'],
                                             initial_lr=params['initial_lr'],
                                             batch_size=params['batch_size'],
                                             num_epochs=params['num_epochs'],
                                             checkpoint_dir=params['checkpoint_dir'],
                                             optimizer_type=params['optimizer_type'],
                                             use_batch_norm=params['use_batch_norm'])

        else:
            raise ValueError('[!] Classifier type {} is not supported.'.format(self.classifier_type))

        if self.classifier_params.has_key('verbose') and self.classifier_params['verbose']:
            print('[*] Initialized a classifier of type {}.'.format(self.classifier_type))

    def get_feature_dir(self):
        """Returns path to feature directory (where features are saved)

        Returns:
            Feature directory.
        """
        return self.classifier_params['feature_dir']

    def get_output_dir(self):
        """Returns path to output directory (where outputs are saved, such as trained classifier, predicted labels, etc.)
        and creates it if it doesn't exist.

        Returns:
            cls_exp_dir: Output directory.

        Raises:
            RuntimeError: If no feature directory was specified in the configuration.
        """
        feature_dir = self.get_feature_dir()
        if feature_dir is None:
            raise RuntimeError('[!] No feature directory or GAN experiment specified.')
        else:
            cls_dir = os.path.join(feature_dir, 'cls')
            cls_exp_dir = os.path.join(cls_dir, self.classifier_params['exp_name'])
            if not os.path.exists(cls_dir):
                os.mkdir(cls_dir)
            if not os.path.exists(cls_exp_dir):
                os.mkdir(cls_exp_dir)
            return cls_exp_dir

    def get_classifier_filename(self):
        """Returns filename for saving the classifier.

        Returns:
            Classifier filename.
        """

        # Classifier filename is parametrized by important experiment parameters.
        return 'classifier_{}_lr{}_rr{}_m{}_c{}_a{}.pkl'.format(self.classifier_params['input_feature'],
                                                                self.classifier_params['learning_rate'],
                                                                self.classifier_params['random_restarts'],
                                                                self.classifier_params['num_measurements'],
                                                                self.classifier_params['counter'],
                                                                self.classifier_params['a_index'])

    def get_labels_filename(self, input_split):
        """Returns filename for saving predicted labels.

        Args:
            input_split: Split to test on [train|val|test].

        Returns:
            Predicted labels filename.
        """

        # Predicted labels filename is parametrized by important experiment parameters.
        return 'predicted_labels_{}_{}_lr{}_rr{}_m{}_c{}_a{}.pkl'.format(input_split,
                                                                         self.classifier_params['input_feature'],
                                                                         self.classifier_params['learning_rate'],
                                                                         self.classifier_params['random_restarts'],
                                                                         self.classifier_params['num_measurements'],
                                                                         self.classifier_params['counter'],
                                                                         self.classifier_params['a_index'])

    def tf_checkpoint_dir(self):
        """Returns name of TensorFlow checkpoint directory.

        Returns:
            Checkpoint directory.
        """

        return 'tf_checkpoints_{}_lr{}_rr{}_m{}_c{}_a{}'.format(self.classifier_params['input_feature'],
                                                                self.classifier_params['learning_rate'],
                                                                self.classifier_params['random_restarts'],
                                                                self.classifier_params['num_measurements'],
                                                                self.classifier_params['counter'],
                                                                self.classifier_params['a_index'])

    def get_acc_filename(self, input_split):
        """Returns filenames for all accuracy files.

        Args:
            input_split: Split to test on [train|val|test].

        Returns:
            acc_filename: The filename for the overall prediction accuracy on this split.
            acc_filenames_i: An array of filenames for class-specific accuracies on this split.
        """

        # Accuracy filename parametrized by experiment parameters.
        acc_filename = 'accuracy_{}_{}_lr{}_rr{}_m{}_c{}_a{}.txt'.format(input_split,
                                                                         self.classifier_params['input_feature'],
                                                                         self.classifier_params['learning_rate'],
                                                                         self.classifier_params['random_restarts'],
                                                                         self.classifier_params['num_measurements'],
                                                                         self.classifier_params['counter'],
                                                                         self.classifier_params['a_index'])
        # For every class, add class number to filename.
        acc_filenames_i = []
        for i in range(self.classifier_params['num_classes']):
            acc_filenames_i.append('class{}_accuracy_{}_{}_lr{}_rr{}_m{}_c{}_a{}.txt'.format(i, input_split,
                                                                                             self.classifier_params[
                                                                                                 'input_feature'],
                                                                                             self.classifier_params[
                                                                                                 'learning_rate'],
                                                                                             self.classifier_params[
                                                                                                 'random_restarts'],
                                                                                             self.classifier_params[
                                                                                                 'num_measurements'],
                                                                                             self.classifier_params[
                                                                                                 'counter'],
                                                                                             self.classifier_params[
                                                                                                 'a_index']))
        return acc_filename, acc_filenames_i

    def train(self, features=None, labels=None, retrain=False, num_train=-1):
        """Trains classifier using training features and ground truth training labels.

        Args:
            features: Path to training feature vectors (use None to automatically load saved features from experiment
            output directory).
            labels: Path to ground truth train labels (use None to automatically load from dataset).
            retrain: Boolean, whether or not to retrain if classifier is already saved.
            num_train: Number of training samples to use (use -1 to include all training samples).

        Raises:
            ValueError: If the specified dataset [mnist|f-mnist|celeba] or classifier type
            [svm|linear-svm|lmnn|logistic|knn|nn] is not supported.
        """

        # If no feature vector is provided load from experiment output directory.
        if features is None:
            feature_file = self.classifier_params['feature_file']
            try:
                with open(feature_file, 'r') as f:
                    features = cPickle.load(f)
            except IOError as err:
                print("[!] I/O error({0}): {1}.".format(err.errno,
                                                        err.strerror))
            if self.classifier_params.has_key('verbose') and self.classifier_params['verbose']:
                print('[*] Loaded feature file from {}.'.format(feature_file))

        # If no label vector is provided load from dataset.
        if labels is None:
            # Create dataset object based on dataset name.
            if self.classifier_params['dataset'] == 'mnist':
                ds = Mnist()
            elif self.classifier_params['dataset'] == 'f-mnist':
                ds = FMnist()
            elif self.classifier_params['dataset'] == 'celeba':
                ds = CelebA(resize_size=self.classifier_params['output_height'],
                            attribute=self.classifier_params['attribute'])
            else:
                raise ValueError('[!] Dataset {} is not supported.'.format(self.classifier_params['dataset']))
            # Load labels from the train split.
            _, labels, _ = ds.load('train')
            num_samples = min(np.shape(features)[0], len(labels))

            # Restrict to the first num_train samples if num_train is not -1.
            if num_train > -1:
                num_samples = min(num_train, num_samples)

            labels = labels[:num_samples]
            features = features[:num_samples, :]

            if self.classifier_params.has_key('verbose') and self.classifier_params['verbose']:
                print('[*] Loaded ground truth labels from {}.'.format(
                    self.classifier_params['dataset']))

        # Train the classifier.
        if self.classifier_type in ('svm', 'logistic', 'knn', 'linear-svm'):
            self.estimator.fit(features, labels)

        # Neural network classifiers.
        elif self.classifier_type == 'nn':
            self.estimator.fit(features, labels, retrain=retrain, session=self.session)

        # For LMNN, first transform the feature vector then perform k-NN.
        elif self.classifier_type == 'lmnn':
            # Learn the metric.
            self.helper_estimator.fit(features, labels)
            # Transform feature space.
            transformed_features = self.helper_estimator.transform(features)
            # Create k-nn graph.
            self.estimator.fit(transformed_features, labels)

        else:
            raise ValueError('[!] Classifier type {} is not supported.'.format(self.classifier_type))

        if ('verbose' in self.classifier_params) and self.classifier_params['verbose']:
            print('[*] Trained classifier.')

    def save_classifier(self, filename=None):
        """Saves the classifier in a pickle file.

        Args:
            filename: Path to pickle file.

        Raises:
            IOError: If a output error occurs while saving the pickle file.
        """

        # If no filename is provided, default filename will be used.
        if filename is None:
            output_dir = self.get_output_dir()
            filename = self.get_classifier_filename()
            filename = os.path.join(output_dir, filename)

        # Saving for non neural-network classifiers.
        if not self.classifier_type == 'nn':
            try:
                with open(filename, 'wb') as fp:
                    cPickle.dump(self.classifier_type, fp, cPickle.HIGHEST_PROTOCOL)
                    cPickle.dump(self.classifier_params, fp, cPickle.HIGHEST_PROTOCOL)
                    cPickle.dump(self.estimator, fp, cPickle.HIGHEST_PROTOCOL)
                    cPickle.dump(self.helper_estimator, fp, cPickle.HIGHEST_PROTOCOL)
            except IOError as err:
                print("[!] I/O error({0}): {1}.".format(err.errno,
                                                        err.strerror))

            if self.classifier_params.has_key('verbose') and self.classifier_params['verbose']:
                print('[*] Saved classifier {}.'.format(filename))

        # Neural network classifiers have default saving/loading using TensorFlow.
        else:
            if self.classifier_params.has_key('verbose') and self.classifier_params['verbose']:
                print('[!] Default TF loading/saving for Neural Networks.')

    def load_classifier(self, filename=None):
        """Loads classifier from a pickle file.

        Args:
            filename: Path to pickle file.

        Raises:
            IOError: If an input error occurs while reading pickle file.
        """

        # If no filename is provided, default filename will be used.
        if filename is None:
            output_dir = self.get_output_dir()
            filename = self.get_classifier_filename()
            filename = os.path.join(output_dir, filename)

        # Loading for non neural-network classifiers.
        if not self.classifier_type == 'nn':
            try:
                with open(filename, 'r') as f:
                    self.classifier_type = cPickle.load(f)
                    self.classifier_params = cPickle.load(f)
                    self.estimator = cPickle.load(f)
                    self.helper_estimator = cPickle.load(f)
            except IOError as err:
                print("[!] I/O error({0}): {1}.".format(err.errno,
                                                        err.strerror))

            if self.classifier_params.has_key('verbose') and self.classifier_params['verbose']:
                print('[*] Loaded classifier from {}.'.format(filename))

        # Neural network classifiers have default saving/loading using TensorFlow.
        else:
            if self.classifier_params.has_key('verbose') and self.classifier_params['verbose']:
                print('[!] Default TF loading/saving for Neural Networks.')

    def predict(self, features, save_result=False, model_name=None, filename=None):
        """Predicts labels given test feature vectors. If save_result is True, also saves the predictions.

        Args:
            features: Test feature vectors.
            save_result: Optional, boolean, if True save predicted labels and accuracy.
            model_name: For neural network classifiers, model name to load and use to predict.
            filename: Optional, path to save results in.

        Returns:
            predicted_labels: Array of predicted labels.

        Raises:
            IOError: If save_result is True and an output error occurs while saving predictions.
            ValueError: If the classifier type is not supported. Supported types: [svm|linear-svm|lmnn|logistic|knn|nn]
        """

        # If save_result is True and no filename was provided, use default filename.
        if save_result and (filename is None):
            output_dir = self.get_output_dir()
            filename = self.get_labels_filename('user_defined')
            filename = os.path.join(output_dir, filename)

        # For kernel and linear SVMs, Logistic regression, and K-NN, simply call the estimator's predict function.
        if self.classifier_type in ('svm', 'logistic', 'knn', 'linear-svm'):
            predicted_labels = self.estimator.predict(features)
            if save_result:
                try:
                    with open(filename, 'wb') as fp:
                        cPickle.dump(predicted_labels, fp, cPickle.HIGHEST_PROTOCOL)
                except IOError as err:
                    print("[!] I/O error({0}): {1}.".format(err.errno,
                                                            err.strerror))
                if self.classifier_params.has_key('verbose') and self.classifier_params['verbose']:
                    print('[*] Saved predicted labels {}.'.format(filename))

            return predicted_labels

        # Same for neural networks, except for the additional model name and TensorFlow session arguments.
        elif self.classifier_type == 'nn':
            predicted_labels = self.estimator.predict(features, model_name, session=self.session)
            if save_result:
                try:
                    with open(filename, 'wb') as fp:
                        cPickle.dump(predicted_labels, fp, cPickle.HIGHEST_PROTOCOL)
                except IOError as err:
                    print("[!] I/O error({0}): {1}.".format(err.errno,
                                                            err.strerror))
                if self.classifier_params.has_key('verbose') and self.classifier_params['verbose']:
                    print('[*] Saved predicted labels {}.'.format(filename))

            return predicted_labels

        # Metric learning.
        elif self.classifier_type == 'lmnn':
            # First transform the features.
            transformed_features = self.helper_estimator.transform(features)
            # Then call the predict function.
            predicted_labels = self.estimator.predict(transformed_features)

            if save_result:
                try:
                    with open(filename, 'wb') as fp:
                        cPickle.dump(predicted_labels, fp, cPickle.HIGHEST_PROTOCOL)
                except IOError as err:
                    print("[!] I/O error({0}): {1}.".format(err.errno,
                                                            err.strerror))
                if self.classifier_params.has_key('verbose') and self.classifier_params['verbose']:
                    print('[*] Saved predicted labels {}.'.format(filename))

            return predicted_labels

        else:
            raise ValueError('[!] Classifier type {} is not supported.'.format(self.classifier_type))

    def validate(self):
        """Only needed for neural networks. Validates different checkpoints by testing them on the validation split and
        retaining the one with the top accuracy.

        Returns:
            best_model: Name of chosen best model (empty string if no validation was performed). An empty string is
            returned for non neural network classifiers.

        Raises:
            IOError: If an input error occurs when loading feature vectors, or an output error occurs when saving the
            chosen model.
            ValueError: If the specified dataset [mnist|f-mnist|celeba] or classifier type
            [svm|linear-svm|lmnn|logistic|knn|nn] is not supported.
        """

        if 'verbose' in self.classifier_params and self.classifier_params['verbose']:
            print("[*] Validating.")

        # Get feature file paths.
        feature_dir = os.path.dirname(self.classifier_params['feature_file'])
        feature_file = os.path.basename(self.classifier_params['feature_file'])
        feature_file = feature_file.replace('train', 'val')
        feature_file = os.path.join(feature_dir, feature_file)

        # Load feature vectors.
        try:
            with open(feature_file, 'r') as f:
                features = cPickle.load(f)
        except IOError as err:
            print("[!] I/O error({0}): {1}.".format(err.errno, err.strerror))

        if 'verbose' in self.classifier_params and self.classifier_params['verbose']:
            print('[*] Loaded feature vectors from {}.'.format(feature_file))

        # Initialize the dataset object to load ground-truth labels.
        if self.classifier_params['dataset'] == 'mnist':
            ds = Mnist()
        elif self.classifier_params['dataset'] == 'f-mnist':
            ds = FMnist()
        elif self.classifier_params['dataset'] == 'celeba':
            ds = CelebA(resize_size=self.classifier_params['output_height'],
                        attribute=self.classifier_params['attribute'])
        else:
            raise ValueError('[!] Dataset {} is not supported.'.format(self.classifier_params['dataset']))

        # Load ground-truth labels from the validation split.
        _, labels, _ = ds.load('val')
        num_samples = min(np.shape(features)[0], len(labels))
        labels = labels[:num_samples]
        features = features[:num_samples, :]

        if 'verbose' in self.classifier_params and self.classifier_params['verbose']:
            print('[*] Loaded ground-truth labels from {}.'.format(
                self.classifier_params['dataset']))

        # Non neural network classifiers do not require validation as no intermediate models exist.
        if self.classifier_type in ('svm', 'logistic', 'knn', 'linear-svm', 'lmnn'):
            print('[!] No validation needed.')
            return ""

        # Neural network classifiers.
        elif self.classifier_type == 'nn':
            # Call the neural network validate function on the features.
            best_acc, best_model, _ = self.estimator.validate(features, labels, session=self.session)

            # Save results.
            try:
                with open(os.path.join(self.get_output_dir(), self.tf_checkpoint_dir(), 'chosen_model.txt'), 'w') as fp:
                    fp.write("{} {}".format(os.path.basename(best_model), best_acc))
            except IOError as err:
                print("[!] I/O error({0}): {1}.".format(err.errno,
                                                        err.strerror))

            if 'verbose' in self.classifier_params and self.classifier_params['verbose']:
                print(
                    '[*] Chose model: {}, with validation accuracy {}.'.format(os.path.basename(best_model), best_acc))
            return best_model

        else:
            raise ValueError('[!] Classifier type {} is not supported.'.format(self.classifier_type))

    def test_classifier(self, input_split='test', save_result=False, model_name=None, labels_filename=None,
                        acc_filename=None, acc_filenames_i=None):
        """Predicts labels and compares them to ground truth labels from given split. Returns test accuracy.
        Args:
            input_split: What split to test on [train|val|test].
            save_result: Optional, boolean. If True saves predicted labels and accuracy.
            model_name:  For neural network classifiers, model name to load and use to predict.
            labels_filename: Optional, string. Path to save predicted labels in.
            acc_filename: Optional, string. Path to save predicted accuracy in.
            acc_filenames_i: Optional, array of strings. Path to save class-specific predicted labels in.

        Returns:
            predicted_labels: Predicted labels for the input split.
            accuracy: Accuracy on the input split.
            per_class_accuracies: Array of per-class accuracies on the input split.

        Raises:
            IOError: If an input error occurs when loading features, or an output error occurs when saving results.
            ValueError: If the specified dataset [mnist|f-mnist|celeba] or classifier type
            [svm|linear-svm|lmnn|logistic|knn|nn] is not supported.
        """

        # If save_result is True, but no labels_filename was specified, use default filename.
        if save_result and (labels_filename is None):
            output_dir = self.get_output_dir()
            labels_filename = self.get_labels_filename(input_split)
            labels_filename = os.path.join(output_dir, labels_filename)

        # If save_result is True, but no acc_filename was specified, use default filename.
        if save_result and (acc_filename is None):
            output_dir = self.get_output_dir()
            acc_filename, acc_filenames_i = self.get_acc_filename(input_split)
            acc_filename = os.path.join(output_dir, acc_filename)
            for i in range(self.classifier_params['num_classes']):
                acc_filenames_i[i] = os.path.join(output_dir, acc_filenames_i[i])

        # Load feature vectors.
        feature_dir = os.path.dirname(self.classifier_params['feature_file'])
        feature_file = os.path.basename(self.classifier_params['feature_file'])
        feature_file = feature_file.replace('train', input_split)
        feature_file = os.path.join(feature_dir, feature_file)

        try:
            with open(feature_file, 'r') as f:
                features = cPickle.load(f)
        except IOError as err:
            print('[!] I/O error({0}): {1}.'.format(err.errno, err.strerror))

        if 'verbose' in self.classifier_params and self.classifier_params['verbose']:
            print('[*] Loaded feature vectors from {}.'.format(feature_file))

        # Initiate dataset object to load ground-truth labels.
        if self.classifier_params['dataset'] == 'mnist':
            ds = Mnist()
        elif self.classifier_params['dataset'] == 'f-mnist':
            ds = FMnist()
        elif self.classifier_params['dataset'] == 'celeba':
            ds = CelebA(resize_size=self.classifier_params['output_height'],
                        attribute=self.classifier_params['attribute'])
        else:
            raise ValueError('[!] Dataset {} is not supported.'.format(self.classifier_params['dataset']))

        # Load ground-truth labels.
        _, labels, _ = ds.load(input_split)
        num_samples = min(np.shape(features)[0], len(labels))
        labels = labels[:num_samples]
        features = features[:num_samples, :]

        if 'verbose' in self.classifier_params and self.classifier_params['verbose']:
            print('[*] Loaded ground-truth labels from: {}.'.format(
                self.classifier_params['dataset']))

        # Predict labels.
        if self.classifier_type in ('svm', 'logistic', 'knn', 'linear-svm', 'lmnn'):
            predicted_labels = self.predict(features, save_result, labels_filename)
        elif self.classifier_type == 'nn':
            predicted_labels = self.predict(features, save_result, model_name, labels_filename)
        else:
            raise ValueError('[!] Classifier type {} is not supported.'.format(self.classifier_type))

        # Compare predicted labels to ground-truth labels and calculate accuracy.
        num_correct = np.sum(np.equal(predicted_labels, labels))
        accuracy = num_correct / (1.0 * len(labels))
        per_class_accuracies = []
        for i in range(self.classifier_params['num_classes']):
            idx = np.where(np.equal(labels, i))[0]
            num_correct = np.sum(np.equal(predicted_labels[idx], labels[idx]))
            accuracy_i = num_correct / (1.0 * len(labels[idx]))
            per_class_accuracies.append(accuracy_i)

        # Save results.
        if save_result:
            try:
                with open(acc_filename, 'w') as fp:
                    fp.write("{}".format(accuracy))
            except IOError as err:
                print("[!] I/O error({0}): {1}.".format(err.errno,
                                                        err.strerror))

            if self.classifier_params.has_key('verbose') and self.classifier_params['verbose']:
                print('[*] Saved predicted labels {}.'.format(labels_filename))
                print('[*] Saved predicted accuracy {}.'.format(acc_filename))

            for i in range(self.classifier_params['num_classes']):
                try:
                    with open(acc_filenames_i[i], 'w') as fp:
                        fp.write("{}".format(per_class_accuracies[i]))
                except IOError as err:
                    print("[!] I/O error({0}): {1}.".format(err.errno,
                                                            err.strerror))

        if self.classifier_params.has_key('verbose') and self.classifier_params['verbose']:
            print('[*] Testing complete. Accuracy on {} split {}.'.format(
                input_split, accuracy))
            for i in range(self.classifier_params['num_classes']):
                print('[*] Testing complete. Accuracy on {} split, class {}: {}.'.format(input_split, i,
                                                                                         per_class_accuracies[i]))

        return predicted_labels, accuracy, per_class_accuracies
