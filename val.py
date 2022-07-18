import numpy as np

from datasets.taxibj import data_permute


def compute_errors(preds, y_true):
    pred_mean = preds[:, 0:2]
    diff = y_true - pred_mean

    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))

    return mse, mae, rmse


def valid(model, val_dataloader, device):
    model.to(device)
    model.eval()
    rmse_list, mse_list, mae_list = [], [], []
    for i, (X_c, X_p, X_t, X_meta, labels) in enumerate(val_dataloader):
        X_c, X_p, X_t, X_meta, labels = data_permute(X_c, X_p, X_t, X_meta, labels, device)
        outputs = model(X_c, X_p, X_t, X_meta)
        mse, mae, rmse = compute_errors(outputs.cpu().data.numpy(), labels.cpu().data.numpy())
        rmse_list.append(rmse)
        mse_list.append(mse)
        mae_list.append(mae)

    rmse = np.mean(rmse_list)
    mse = np.mean(mse_list)
    mae = np.mean(mae_list)

    return rmse, mse, mae
