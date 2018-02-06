import os
import yaml


def get_cls_param_dict(cfg_file):
    """Creates a dictionary of parameters based on configuration file.

    Args:
        cfg_file: Path to classifier configuration file.

    Returns:
        param_dict: Classifier parameter dictionary.

    Raises:
        IOError: If an input error occurs when reading the configuration file.
    """

    try:
        with open(cfg_file, 'r') as f:
            cfg = yaml.load(f)
    except IOError as err:
        print "I/O error({0}): {1}".format(err.errno, err.strerror)

    param_dict = dict()
    param_dict['counter'] = None

    for (k, v) in cfg.items():
        param_dict[k.lower()] = v['val']  # Put all configuration params into dictionary.

    if 'exp_name' not in param_dict:
        exp_name = os.path.splitext(os.path.basename(cfg_file))[0]
        param_dict['exp_name'] = exp_name

    return param_dict
