import argparse

import torch

from utils.util import load_trainvaltest_dataloader, load_conf
from val import valid


def test(model, test_dataloader, mmn, device):
    return valid(model, test_dataloader, mmn, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--conf_path', type=str, default='configs/taxibj_local.yaml', help='conf path')
    parser.add_argument('--model_path', type=str, default='workdir/best.pth', help='model path')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conf = load_conf(args.conf_path)
    model = torch.load(args.model_path, map_location=device)['model']
    _, _, test_dataloader, mmn = load_trainvaltest_dataloader(conf)
    rmse, mse, mae = test(model, test_dataloader, mmn, device)
    print('Test, rmse: {}, mse: {}, mae: {}'.format(rmse, mse, mae))
