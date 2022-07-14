import torch
import torch.utils.data as data

from datasets.dataloader import load_dataset

T = 24
days_test = 10
len_test = T * days_test


def data_permute(X_c, X_p, X_t, X_meta, Y_batch, device):
    X_c = X_c.type(torch.FloatTensor).to(device)
    X_p = X_p.type(torch.FloatTensor).to(device)
    X_t = X_t.type(torch.FloatTensor).to(device)
    X_meta = X_meta.type(torch.FloatTensor).to(device)
    Y_batch = Y_batch.type(torch.FloatTensor).to(device)

    X_c = X_c.permute(0, 3, 1, 2)
    X_p = X_p.permute(0, 3, 1, 2)
    X_t = X_t.permute(0, 3, 1, 2)
    Y_batch = Y_batch.permute(0, 3, 1, 2)

    return X_c, X_p, X_t, X_meta, Y_batch


class TaxiBJ(data.Dataset):
    def __init__(self, data_root, cache_path, dataset_name, mode, len_closeness, len_period, len_trend):
        self.data_root = data_root
        self.cache_path = cache_path
        self.dataset_name = dataset_name
        self.mode = mode
        self.len_closeness = len_closeness
        self.len_period = len_period
        self.len_trend = len_trend

        if self.dataset_name == 'TaxiBJ':
            print("loading data...")

            if self.mode == 'train':
                self.X_data, self.Y_data, _, _, mmn, external_dim, timestamp_train, timestamp_test = load_dataset(
                    data_root=self.data_root,
                    cache_path=self.cache_path,
                    len_closeness=self.len_closeness,
                    len_period=self.len_period,
                    len_trend=self.len_trend,
                    len_test=len_test,
                    preprocess_name='preprocessing.pkl',
                    meta_data=True)

            elif self.mode == 'test':
                _, _, self.X_data, self.Y_data, mmn, external_dim, timestamp_train, timestamp_test = load_dataset(
                    data_root=self.data_root,
                    cache_path=self.cache_path,
                    len_closeness=self.len_closeness,
                    len_period=self.len_period,
                    len_trend=self.len_trend,
                    len_test=len_test,
                    preprocess_name='preprocessing.pkl',
                    meta_data=True)

            assert len(self.X_data[0]) == len(self.Y_data)
            self.data_len = len(self.Y_data)

        else:
            print('Unknown datasets')

        self.mmn = mmn

    def __len__(self):
        'Denotes the total number of samples'
        return self.data_len

    def __str__(self):
        string = '' \
                 + '\tmode   = %s\n' % self.mode \
                 + '\tdataset name   = %s\n' % self.dataset_name \
                 + '\tmmn min   = %d\n' % self.mmn._min \
                 + '\tmmn max   = %d\n' % self.mmn._max \
                 + '\tlen    = %d\n' % len(self)

        return string

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X_c, X_p, X_t, X_meta = self.X_data[0][index], self.X_data[1][index], self.X_data[2][index], self.X_data[3][
            index]
        y = self.Y_data[index]

        return X_c, X_p, X_t, X_meta, y
