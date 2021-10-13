import pandas as pd
import numpy as np

def get_training_data(
    indices_path, gim_path, gim_meta_path,
    indices = ['Kp', 'ap', 'f10.7', 'R', 'AE', 'AL', 'AU'],
    log_indices = True,
    sqrt_indices = True, 
    lookback = [24],
    lookback_windowed = [12],
    periods = ['24h', '366d', '183d', '27d'],
    validation_year=2017,
    test_year=2018,
):

    maps_meta = pd.read_csv(gim_meta_path).reset_index()
    maps = np.load(gim_path)

    meta = pd.read_csv(indices_path)
    meta = meta[['datetime'] + indices]
    meta = meta.fillna(0)
    meta = meta.merge(maps_meta, on='datetime', how='left')
    meta = meta.rename({'index': 'map_index'}, axis=1)

    new_indices = [i for i in indices]
    columns_wo_negatives = (meta[indices] >= 0).all()
    columns_wo_negatives = columns_wo_negatives.index[columns_wo_negatives]
    if log_indices:
        logged_meta = np.log(meta[columns_wo_negatives] + 1)
        logged_meta.columns = [column + '_log' for column in logged_meta.columns]
        new_indices = new_indices + list(logged_meta.columns)
        meta = pd.concat([meta, logged_meta], axis=1)
    if sqrt_indices:
        sqrt_meta = np.sqrt(meta[columns_wo_negatives])
        sqrt_meta.columns = [column + '_sqrt' for column in sqrt_meta.columns]
        new_indices = new_indices + list(sqrt_meta.columns)
        meta = pd.concat([meta, sqrt_meta], axis=1)
    indices = new_indices

    meta.index = pd.to_datetime(meta.datetime)
    meta = meta.drop(columns=['datetime'])
    meta = meta.resample('1h').mean()

    for shift in lookback:
        meta_shifted = meta[indices].shift(shift)
        meta_shifted.columns = [column + '_' + str(shift) for column in meta_shifted.columns]
        meta = pd.concat([meta, meta_shifted], axis=1)
    for window in lookback_windowed:
        meta_mean = meta[indices].rolling(window).mean()
        meta_mean.columns = [column + '_m' + str(window) for column in meta_mean.columns]
        meta_std = meta[indices].rolling(window).std()
        meta_std.columns = [column + '_s' + str(window) for column in meta_std.columns]
        meta = pd.concat([meta, meta_mean, meta_std], axis=1)

    meta = meta.reset_index()
    meta = meta.assign(hours = meta.datetime.view('int') / (10 ** 9) / 3600)
    for period in periods:
        unit = period[-1]
        value = int(period[:-1])
        if unit == 'h':
            normalized = (meta.hours % value) / value * 2 * np.pi
        if unit == 'd':
            normalized = (meta.hours % value) / value * 2 * np.pi / 24
        period_sin = np.sin(normalized)
        period_cos = np.cos(normalized)
        periods_df = pd.DataFrame({
            period + '_sin': period_sin,
            period + '_cos': period_cos,
        })
        meta = pd.concat([meta, periods_df], axis=1)

    meta = meta.dropna(how='any', axis=0)

    train_meta = meta[
        (meta.datetime.dt.year < validation_year) & 
        (meta.datetime.dt.year < test_year)
    ]
    train_maps = maps[train_meta.map_index.astype(int).tolist()]
    val_meta = meta[
        (meta.datetime.dt.year == validation_year)
    ]
    val_maps = maps[val_meta.map_index.astype(int).tolist()]
    test_meta = meta[
        (meta.datetime.dt.year == test_year)
    ]
    test_maps = maps[test_meta.map_index.astype(int).tolist()]

    train_meta = train_meta.drop(columns=['datetime', 'hours', 'map_index'])
    val_meta = val_meta.drop(columns=['datetime', 'hours', 'map_index'])  
    test_meta = test_meta.drop(columns=['datetime', 'hours', 'map_index'])

    return (train_meta, train_maps), (val_meta, val_maps), (test_meta, test_maps)