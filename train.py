import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

from datasets.taxibj import TaxiBJ, data_permute
from st_resnet import STResNet
from val import valid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# configs
random_seed = 42

network_conf = {
    'len_closeness': 3,
    'len_period': 1,
    'len_trend': 1,
    'external_dim': 28,
    'repeat_num': 4,
}

data_conf = {
    'name': 'TaxiBJ',
    'root_path': '/content/drive/MyDrive/datasets/TaxiBJ',
    'cache_path': '/content/drive/MyDrive/datasets/TaxiBJ/cache',
    'shuffle': True,
    'flow': 2,
    'height': 32,
    'width': 32,
}

train_conf = {
    'val_split': 0.1,
    'max_epoch': 500,
    'lr': 0.0002,
    'batch_size': 500,
    'shuffle': False,
    'drop_last': False,
    'num_workers': 0,
    'save_interval': 5,
    'save_dir': '/content/drive/MyDrive/projects/ST_ResNet/workdir',
}

# 随机划分训练集和验证集
taxibj_dataset = TaxiBJ(data_root=data_conf['root_path'], cache_path=data_conf['cache_path'],
                        dataset_name=data_conf['name'], mode='train',
                        len_closeness=network_conf['len_closeness'], len_period=network_conf['len_period'],
                        len_trend=network_conf['len_trend'])
dataset_size = len(taxibj_dataset)
indices = list(range(dataset_size))
split = int(np.floor(train_conf['val_split'] * dataset_size))
if data_conf['shuffle']:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
train_dataloader = data.DataLoader(
    taxibj_dataset,
    batch_size=train_conf['batch_size'],
    shuffle=train_conf['shuffle'],
    drop_last=train_conf['drop_last'],
    num_workers=train_conf['num_workers'],
    sampler=train_sampler,
)
valid_dataloader = data.DataLoader(
    taxibj_dataset,
    batch_size=train_conf['batch_size'],
    shuffle=train_conf['shuffle'],
    drop_last=train_conf['drop_last'],
    num_workers=train_conf['num_workers'],
    sampler=valid_sampler,
)
print('train size: {}, valid size: {}'.format(len(train_indices), len(val_indices)))

# model
model = STResNet(
    (network_conf['len_closeness'], data_conf['flow'], data_conf['height'], data_conf['width']),
    (network_conf['len_period'], data_conf['flow'], data_conf['height'], data_conf['width']),
    (network_conf['len_trend'], data_conf['flow'], data_conf['height'], data_conf['width']),
    external_dim=network_conf['external_dim'],
    repeat_num=network_conf['repeat_num']
)
print(model)

# train setting
loss_mse = nn.MSELoss()  # nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=train_conf['lr'])

if not os.path.exists(train_conf['save_dir']):
    os.mkdir(train_conf['save_dir'])

model.to(device)
loss_mse.to(device)

best_rmse, best_epoch = 1., 0
for epoch in range(train_conf['max_epoch']):
    for i_iter, (X_c, X_p, X_t, X_meta, Y_batch) in enumerate(train_dataloader):
        X_c, X_p, X_t, X_meta, Y_batch = data_permute(X_c, X_p, X_t, X_meta, Y_batch, device)

        # Forward pass
        outputs = model(X_c, X_p, X_t, X_meta)
        loss = loss_mse(outputs, Y_batch)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # val and print
        if i_iter % 10 == 0:
            print('TRAIN, epoch: {}/{}, iter: {}, loss: {}'.format(epoch, train_conf['max_epoch'], i_iter, loss.item()))

    rmse, mse, mae = valid(model, valid_dataloader, device)
    print('VAL, epoch: {}, rmse: {}, mse: {}, mae: {}'.format(epoch, rmse, mse, mae))

    if rmse < best_rmse:
        best_rmse, best_epoch = rmse, epoch
        torch.save(model.state_dict(), os.path.join(train_conf['save_dir'], 'epoch{}_rmse{}.pth'.format(epoch, rmse)))
        print('VAL, epoch: {}, best_rmse: {}'.format(epoch, best_rmse))

    if epoch % train_conf['save_interval'] == 0:
        torch.save({
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'param': model.state_dict()
        }, os.path.join(train_conf['save_dir'], 'epoch_{}.pth'.format(epoch)))
