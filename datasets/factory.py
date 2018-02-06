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

"""Creates dataset objects."""

import tensorflow as tf

from datasets.celeba import CelebA
from datasets.fmnist import FMnist
from datasets.mnist import Mnist


def load_ds(flags=None):
    """Constructs the dataset that is set in flags.
    
    Args:
        flags: A FLAGS object with properties. If it's not set, use the global flags.
        
    Returns:
        The Dataset object that is set in the flags.
    """

    if flags is None:  # Load the default flags.
        flags = tf.app.flags.FLAGS

    if flags.dataset.lower() == 'mnist':
        ds = Mnist()
    elif flags.dataset.lower() == 'f-mnist':
        ds = FMnist()
    elif flags.dataset.lower() == 'celeba':
        if hasattr(flags, 'attribute'):
            ds = CelebA(resize_size=flags.output_height, attribute=flags.attribute)
        else:
            ds = CelebA(resize_size=flags.output_height)
    else:
        raise ValueError('[!] Dataset {} is not supported.'.format(flags.dataset.lower()))
    return ds
