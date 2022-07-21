import argparse
import os

import numpy as np
import torch

from datasets.taxibj import data_permute
from models.st_resnet import STResNet
from test import test
from utils.util import load_conf, load_trainvaltest_dataloader, load_loss, EarlyStopping, update_latest
from val import valid

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--conf_path', type=str, default='configs/taxibj_config.yaml', help='conf path')
args = parser.parse_args()

# load config
conf = load_conf(args.conf_path)

# get train/val/test loader
train_dataloader, valid_dataloader, test_dataloader, mmn = load_trainvaltest_dataloader(conf)

# load model and loss
if os.path.exists(conf['training']['resume_path']):
    resume = torch.load(conf['training']['resume_path'])
    model, optimizer, start_epoch = resume['model'], resume['optimizer'], resume['epoch'] + 1
    print('load model, optimizer, epoch from {}'.format(conf['training']['resume_path']))
else:
    model = STResNet(
        (conf['task']['len_closeness'], conf['dataset']['flow'], conf['dataset']['height'], conf['dataset']['width']),
        (conf['task']['len_period'], conf['dataset']['flow'], conf['dataset']['height'], conf['dataset']['width']),
        (conf['task']['len_trend'], conf['dataset']['flow'], conf['dataset']['height'], conf['dataset']['width']),
        conf['task']['external_dim'],
        conf['network']['repeat_num']
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=conf['training']['lr'])
    start_epoch = 0

loss_mse = load_loss(conf)

# load training setting
if not os.path.exists(conf['training']['save_dir']):
    os.mkdir(conf['training']['save_dir'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
loss_mse.to(device)

# training
best_rmse, best_epoch = 1., 0
es = EarlyStopping(model, os.path.join(conf['training']['save_dir'], 'best.pth'))
for epoch in range(start_epoch, conf['training']['max_epoch']):
    losses = []
    for i_iter, (X_c, X_p, X_t, X_meta, Y_batch) in enumerate(train_dataloader):
        X_c, X_p, X_t, X_meta, Y_batch = data_permute(X_c, X_p, X_t, X_meta, Y_batch, device)

        outputs = model(X_c, X_p, X_t, X_meta)
        loss = loss_mse(outputs, Y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    print('TRAIN, epoch: {}/{}, loss: {}'.format(epoch, conf['training']['max_epoch'], np.mean(losses)))

    # val
    rmse, mse, mae = valid(model, valid_dataloader, mmn, device)
    print('VAL, epoch: {}, rmse: {}, mse: {}, mae: {}'.format(epoch, rmse, mse, mae))

    if es.step(mse):
        print('early stopped! With val loss:', mse)
        break

    if epoch % conf['training']['save_interval'] == 0:
        update_latest(model, optimizer, epoch, os.path.join(conf['training']['save_dir'], 'latest.pth'))

# test best_model
best_model = torch.load(os.path.join(conf['training']['save_dir'], 'best.pth'))['model']
test_rmse, test_mse, test_mae = test(best_model, test_dataloader, mmn, device)
print("TEST, rmse: {}".format(test_rmse))
