import numpy as np

def mae(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    return np.abs(y_pred - y_true)   

def mse(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    return (y_pred - y_true) ** 2   

def rmse(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    return np.sqrt((y_pred - y_true) ** 2)

def mape(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    return 100 * np.abs(y_true - y_pred) / y_true

def smape(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    return 100 * np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2)

def mean_error(y_pred, y_true, error_func):
    error = error_func(y_pred, y_true)
    return error.mean()

def mean_error_by_maps(y_pred, y_true, error_func):
    error = error_func(y_pred, y_true)
    return error.reshape(len(y_pred), -1).mean(axis=1)

def mean_error_by_points(y_pred, y_true, error_func):
    error = error_func(y_pred, y_true)
    return error.mean(axis=0)