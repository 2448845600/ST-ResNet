import os

import numpy as np
import torch
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


def update_latest(model, optimizer, epoch, save_path):
    torch.save({'model': model, 'optimizer': optimizer, 'epoch': epoch}, save_path)


class EarlyStopping(object):
    def __init__(self, model, save_path, mode='min', min_delta=0, patience=30, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)
        self.model = model
        self.save_path = save_path

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics, epoch):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            torch.save({'model': self.model, 'mse': self.best.item(), 'epoch': epoch}, self.save_path)
            print('best model saved at {}'.format(self.save_path))
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                        best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                        best * min_delta / 100)
