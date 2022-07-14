import numpy as np
import torch
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

from datasets.taxibj import TaxiBJ
from st_resnet import STResNet
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# configs
len_closeness = 3  # length of closeness dependent sequence
len_period = 4  # length of peroid dependent sequence
len_trend = 4  # length of trend dependent sequence
repeat_num = 4  # number of residual units
random_seed = 42
validation_split = 0.1
# early_stop_patience = 30
shuffle_dataset = True
max_epoch = 500
map_height, map_width = 16, 8  # grid size
nb_flow = 2  # there are two types of flows: new-flow and end-flow
# nb_area = 81
# m_factor = math.sqrt(1. * map_height * map_width / nb_area)
# print('factor: ', m_factor)
learning_rate = 0.0002
data_root='/content/drive/MyDrive/datasets/TaxiBJ'
cache_path='/content/drive/MyDrive/datasets/TaxiBJ/cache'

params = {
    'batch_size': 500,
    'shuffle': False,
    'drop_last': False,
    'num_workers': 0
}

# dataloader
train_dataset = TaxiBJ(data_root=data_root, cache_path=cache_path, dataset_name='TaxiBJ', mode='train',
                       len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
train_dataloader = data.DataLoader(train_dataset, **params)

# model
model = STResNet(
    (len_closeness, nb_flow, map_height, map_width),
    (len_period, nb_flow, map_height, map_width),
    (len_trend, nb_flow, map_height, map_width),
    external_dim=8,
    repeat_num=repeat_num
)

# train setting
loss_fn = nn.MSELoss() # nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.to(device)
loss_fn.to(device)
for e in range(max_epoch):
    for i, (X_c, X_p, X_t, X_meta, Y_batch) in enumerate(train_dataloader):
        # Move tensors to the configured device
        X_c = X_c.type(torch.FloatTensor).to(device)
        X_p = X_p.type(torch.FloatTensor).to(device)
        X_t = X_t.type(torch.FloatTensor).to(device)
        X_meta = X_meta.type(torch.FloatTensor).to(device)
        # print(X_meta[0])
        Y_batch = Y_batch.type(torch.FloatTensor).to(device)

        # Forward pass
        outputs = model(X_c, X_p, X_t, X_meta)
        # print(outputs[0])
        loss = loss_fn(outputs, Y_batch)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.cpu())
        #save best model