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

"""Contains the configuration handling code and default experiment parameters."""

from types import NoneType
import yaml
import tensorflow as tf
import os

define_funcs = {int: tf.app.flags.DEFINE_integer,
                float: tf.app.flags.DEFINE_float,
                bool: tf.app.flags.DEFINE_boolean,
                basestring: tf.app.flags.DEFINE_string,
                str: tf.app.flags.DEFINE_string,
                type(None): tf.app.flags.DEFINE_integer,
                tuple: tf.app.flags.DEFINE_list,
                list: tf.app.flags.DEFINE_list}


def set_default_params():
    """Sets the default parameters per dataset in the global flags."""
    flags = tf.app.flags.FLAGS
    if flags.dataset.lower() == 'mnist':
        flags.cs_num_random_restarts = 10
        flags.cs_max_update_iter = 100
        flags.cs_learning_rate = 1.0
        flags.cs_decay_lr_iter = 101
    elif flags.dataset.lower() == 'f-mnist':
        flags.cs_num_random_restarts = 10
        flags.cs_max_update_iter = 100
        flags.cs_learning_rate = 0.1
        flags.cs_decay_lr_iter = 101
    elif flags.dataset.lower() == 'celeba':
        flags.cs_num_random_restarts = 2
        flags.cs_max_update_iter = 500
        flags.cs_learning_rate = 0.1
        flags.cs_decay_lr_iter = 800
    else:
        flags.cs_num_random_restarts = 10
        flags.cs_max_update_iter = 100
        flags.cs_learning_rate = 1.0
        flags.cs_decay_lr_iter = 101


def init_flags():
    """Initializes the flags that are None."""

    flags = tf.app.flags.FLAGS

    if flags.input_width is None:
        flags.input_width = flags.input_height
    if flags.output_width is None:
        flags.output_width = flags.output_height
    flags.n_input = flags.output_width * flags.output_height * flags.c_dim


def load_config(cfg_path, set_flag=False, verbose=False):
    """Loads the configuration files into the global flags.
    
    Args:
        cfg_path: The path to the config yaml file. 
        set_flag: If True, does not create new flag attributes, only sets existing ones.
        verbose: Verbose mode.

    Returns:
        A the loaded configuration dictionary.

    Raises:
        RuntimeError: If the configuration path does not exist.
    """
    flags = tf.app.flags.FLAGS

    if not os.path.exists(cfg_path):
        raise RuntimeError(
            "[!] Configuration path {} does not exist.".format(cfg_path))
    if os.path.isdir(cfg_path):
        cfg_path = os.path.join(cfg_path, 'cfg.yml')

    with open(cfg_path, 'r') as f:
        loaded_cfg = yaml.load(f)
    with open('experiments/cfgs/default_cfg.yml', 'r') as f:
        cfg = yaml.load(f)

    cfg.update(loaded_cfg)
    if not 'EXP_NAME' in cfg.keys():
        cfg['EXP_NAME'] = {'val': cfg_path.split('/')[-1][:-4],
                           'doc': 'exp name'}

    tf.app.flags.DEFINE_string('cfg_path', cfg_path, 'config path.')
    # setattr(flags, 'cfg_path', cfg_path)
    for (k, v) in cfg.items():
        if set_flag:
            setattr(flags, k.lower(), v['val'])
        else:
            if hasattr(flags, k.lower()):
                setattr(flags, k.lower(), v['val'])
            else:
                def_func = define_funcs[type(v['val'])]
                def_func(k.lower(), v['val'], v['doc'])
        if verbose:
            print('[#] set {} to {} type: {}'.format(k.lower(), v['val'],
                                                     str(type(
                                                         v['val']))))

    return cfg
