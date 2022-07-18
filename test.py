import argparse

import torch

from utils.util import load_trainvaltest_dataloader, load_conf
from val import valid


def test(model, test_dataloader, device):
    return valid(model, test_dataloader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('conf_path', type=str, help='conf path')
    parser.add_argument('model_path', type=str, help='model path')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conf = load_conf(args.conf_path)
    model = torch.load(args.model_path)
    _, _, test_dataloader = load_trainvaltest_dataloader(conf)
    test(model, test_dataloader, device)
