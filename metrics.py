import numpy
import torch
from sklearn.metrics import mean_squared_error, roc_auc_score


def get_rmse(y, y_hat):
    return numpy.sqrt(mean_squared_error(y, y_hat))


def get_roc_auc(y, y_hat):
    y_hat = y_hat.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    return roc_auc_score(y, y_hat)


def get_pearson_corr(y, y_hat):
    y_hat = y_hat.flatten()
    y = y.flatten()

    y_hat_mean = torch.mean(y_hat)
    y_mean = torch.mean(y)

    covariance = torch.sum((y_hat - y_hat_mean) * (y - y_mean))

    y_hat_std = torch.sqrt(torch.sum((y_hat - y_hat_mean) ** 2))
    y_std = torch.sqrt(torch.sum((y - y_mean) ** 2))

    pearson_corr = covariance / (y_hat_std * y_std)

    return pearson_corr.item()
