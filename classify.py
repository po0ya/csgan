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

"""Train, validate and test a classifier."""

import argparse
import sys
import tensorflow as tf

from models import classification as cls


def parse_args():
    """Parses command-line arguments.

    Returns:
        args: Command-line arguments
    """
    parser = argparse.ArgumentParser()

    # Classification configuration file and feature file are the only two required arguments.
    parser.add_argument('--cfg', required=True, help='Classifier config file')
    parser.add_argument('--feature_file', required=True, help='Feature file to train on.')

    # Optional flags.
    parser.add_argument('--test_split', default='test', required=False, help='Split to test on.')
    parser.add_argument('--retrain', default=False, required=False, help='Whether or not to re-train the classifier'
                                                                         '(for neural networks)',
                        action='store_true')
    parser.add_argument('--validate', default=False, required=False, help='Whether or not to use validation split to'
                                                                          'select the best model (for neural networks)',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args, _ = parser.parse_known_args()
    return args


def main():
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as session:
        # Create classifier from config.
        classifier = cls.Classifier(args.cfg, args.feature_file, args.test_split)
        classifier.session = session
        # Train classifier.
        classifier.train(retrain=args.retrain)

        # Find the best model based on validation set accuracy.
        if args.validate:
            best_model = classifier.validate()
        else:
            best_model = None

        # Only needed for non neural network classifiers.
        classifier.save_classifier()

        # Test classifier on train split.
        _, train_acc, per_class = classifier.test_classifier(input_split='train', save_result=True,
                                                             model_name=best_model)
        # Test classifier on test split.
        _, test_acc, per_class = classifier.test_classifier(input_split=args.test_split, save_result=True,
                                                            model_name=best_model)


if __name__ == '__main__':
    args = parse_args()
    main()
