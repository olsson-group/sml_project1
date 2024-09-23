import numpy
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, roc_auc_score


def get_rmse(y, y_hat):
    y = y.detach().cpu().numpy()
    y_hat = y_hat.detach().cpu().numpy()
    return numpy.sqrt(mean_squared_error(y, y_hat))


def get_auc_roc(y, y_hat):
    y = y.detach().cpu().numpy()
    y_hat = y_hat.detach().cpu().numpy()
    return roc_auc_score(y, y_hat)


def get_pearson_corr(y, y_hat):
    y_hat = y_hat.flatten()
    y = y.flatten()
    return pearsonr(y_hat, y)
