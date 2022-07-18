import os

import torch.nn as nn
import yaml

from datasets.taxibj import get_taxtbj_dataset


def load_conf(conf_path, args=None):
    if os.path.exists(conf_path):
        with open(conf_path) as f:
            conf = yaml.safe_load(f)
    else:
        raise "conf path {} is not exists".format(conf_path)

    if args:
        for k, v in args:
            conf[k] = v
    return conf


def load_trainvaltest_dataloader(conf):
    if conf['dataset']['name'] == 'TaxiBJ':
        return get_taxtbj_dataset(conf)
    else:
        raise "Unknown dataset"


def load_loss(conf):
    if conf['network']['loss'] == 'mse':
        return nn.MSELoss()
    elif conf['network']['loss'] == 'l1':
        return nn.L1Loss()
    else:
        raise "Unknown loss"
