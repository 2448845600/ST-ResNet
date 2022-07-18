import argparse
import os

import torch

from datasets.taxibj import data_permute
from models.st_resnet import STResNet
from test import test
from utils.util import load_conf, load_trainvaltest_dataloader, load_loss
from val import valid

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('conf_path', type=str, help='conf path')
args = parser.parse_args()

# load config
conf = load_conf(args.conf_path)

# get train/val/test loader
train_dataloader, valid_dataloader, test_dataloader = load_trainvaltest_dataloader(conf)

# load model and loss
model = STResNet(
    (conf['task']['len_closeness'], conf['dataset']['flow'], conf['dataset']['height'], conf['dataset']['width']),
    (conf['task']['len_period'], conf['dataset']['flow'], conf['dataset']['height'], conf['dataset']['width']),
    (conf['task']['len_trend'], conf['dataset']['flow'], conf['dataset']['height'], conf['dataset']['width']),
    conf['task']['external_dim'],
    conf['network']['repeat_num']
)
loss_mse = load_loss(conf)

# load training setting
optimizer = torch.optim.Adam(model.parameters(), lr=conf['training']['lr'])
if not os.path.exists(conf['training']['save_dir']):
    os.mkdir(conf['training']['save_dir'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
loss_mse.to(device)

# training
best_rmse, best_epoch = 1., 0
for epoch in range(conf['training']['max_epoch']):
    for i_iter, (X_c, X_p, X_t, X_meta, Y_batch) in enumerate(train_dataloader):
        X_c, X_p, X_t, X_meta, Y_batch = data_permute(X_c, X_p, X_t, X_meta, Y_batch, device)

        # Forward pass
        outputs = model(X_c, X_p, X_t, X_meta)
        loss = loss_mse(outputs, Y_batch)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # val
        if i_iter % 10 == 0:
            print('TRAIN, epoch: {}/{}, iter: {}, loss: {}'.format(
                epoch, conf['training']['max_epoch'], i_iter, loss.item()))

    rmse, mse, mae = valid(model, valid_dataloader, device)
    print('VAL, epoch: {}, rmse: {}, mse: {}, mae: {}'.format(epoch, rmse, mse, mae))

    if rmse < best_rmse:
        best_rmse, best_epoch = rmse, epoch
        torch.save({'model': model, 'rmse': rmse, 'epoch': epoch},
                   os.path.join(conf['training']['save_dir'], 'best.pth'))
        print('VAL, epoch: {}, best_rmse: {}'.format(epoch, best_rmse))

    if epoch % conf['training']['save_interval'] == 0:
        torch.save({'optimizer': optimizer.state_dict(), 'epoch': epoch, 'model': model},
                   os.path.join(conf['training']['save_dir'], 'latest.pth'))

# test best_model
best_model = torch.load(os.path.join(conf['training']['save_dir'], 'best.pth'))
test_rmse, test_mse, test_mae = test(best_model, test_dataloader, device)
print("TEST, rmse: {}".format(test_rmse))
