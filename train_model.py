import os
import numpy as np
import pandas as pd
import mlflow

from errors import mae, mse, rmse, mape, smape
from errors import mean_error, mean_error_by_maps, mean_error_by_points
from data_loaders.prepare_data import get_training_data
from Models import dense_model
from config import gim_path, gim_meta_path, indices_path
from keras import callbacks
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', dest='experiment', type=str, default='Default')
parser.add_argument('--model', dest='model', choices=['NN', 'GIMLinear', 'GIMLi-xgbdt'])
parser.add_argument('--shift', dest='shift', type=int, default=0)

parser.add_argument('--indices', dest='indices', nargs='+', default=[], help="['Kp', 'f10.7', 'R', 'ap', 'AE', 'AL', 'AU']")
parser.add_argument('--log_indices', dest='log_indices', action='store_true')
parser.add_argument('--sqrt_indices', dest='sqrt_indices', action='store_true')

parser.add_argument('--lookback', dest='lookback', nargs='+', default=[]) # Single values from previous records
parser.add_argument('--lookback_windowed', dest='lookback_windowed', nargs='+', default=[]) # Mean values from history
parser.add_argument('--periods', dest='periods', nargs='+', default=[]) # Datetime

parser.add_argument('--validation_year', dest='validation_year', type=int, default=2017)
parser.add_argument('--test_year', dest='test_year', type=int, default=2018)

parser.add_argument('--dropout', dest='dropout', type=float, default=None)
parser.add_argument('--learning-rate', dest='learning_rate', type=float, default=0.001)
parser.add_argument('--epochs', dest='epochs', type=int, default=1)

parser.add_argument('--example-datetimes', dest='example_datetimes', nargs='+', default=['2018-07-15 14:00:00'])

parser.add_argument('--log-errors', dest='log_errors', nargs='+', default=['mae', 'mse', 'rmse', 'mape', 'smape'])
parser.add_argument('--save-prediction', dest='save_prediction', action='store_true')


args = parser.parse_args()

errors = {'mae': mae, 'mse': mse, 'rmse': rmse}
errors_scales = {'mae': 15, 'mse': 225, 'rmse': 15, 'mape': 100, 'smape': 200}

model = args.model
shift = args.shift
mlflow.set_experiment(args.experiment)

with mlflow.start_run():
    indices = args.indices
    log_indices = args.log_indices
    sqrt_indices = args.sqrt_indices

    lookback = [int(i) for i in args.lookback]
    lookback_windowed = [int(i) for i in args.lookback_windowed]
    periods = args.periods

    validation_year = args.validation_year
    test_year = args.test_year

    save_prediction = args.save_prediction

    if model == 'NN':
        dropout = args.dropout
        learning_rate = args.learning_rate
        epochs = args.epochs

    example_datetimes = args.example_datetimes

    log_errors = args.log_errors
    errors = {key:value for key, value in errors.items() if key in log_errors}

    mlflow.log_params({
        'indices': indices,
        'log_indices': log_indices,
        'sqrt_indices': sqrt_indices,
        'lookback': lookback,
        'lookback_windowed': lookback_windowed,
        'periods': periods,
        'validation_year': validation_year,
        'test_year': test_year,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'shift': shift,
    })

    (train_meta, train_maps), \
        (val_meta, val_maps), \
        (test_meta, test_maps), meta = get_training_data(
            indices_path, gim_path, gim_meta_path,
            indices=indices,
            log_indices=log_indices,
            sqrt_indices=sqrt_indices,
            shift=shift,
            lookback=lookback,
            lookback_windowed=lookback_windowed,
            periods=periods,
            validation_year=validation_year,
            test_year=test_year,

        )

    print(train_meta.shape)
    print(test_meta.shape)
    print(val_meta.shape)   

    if os.path.isdir('/tmp/training_data'):
        os.system('rm -rf /tmp/training_data')
    os.mkdir('/tmp/training_data')
    meta.to_csv('/tmp/training_data/meta.csv', index=False)
    train_meta.to_csv('/tmp/training_data/train_meta.csv', index=False)
    val_meta.to_csv('/tmp/training_data/val_meta.csv', index=False)
    test_meta.to_csv('/tmp/training_data/test_meta.csv', index=False)    

    mlflow.log_artifact('/tmp/training_data')

    model = dense_model.create_model(train_meta.drop(columns=['datetime']).values[0].shape)
    
    history = model.fit(
        train_meta.drop(columns=['datetime']).values, train_maps,
        validation_data=(val_meta.drop(columns=['datetime']).values, val_maps),
        epochs=epochs,
        callbacks=[callbacks.ModelCheckpoint('/tmp/weights.h5', save_best_only=True, monitor='val_loss')],
    )
    model.load_weights('/tmp/weights.h5')
    model.save('/tmp/model')

    mlflow.keras.log_model(model, 'dense_model')
    for i in range(epochs):
        mlflow.log_metrics({metric: values[i] for metric, values in history.history.items()})

    train_generated_maps = model.predict(train_meta.drop(columns=['datetime']).values)
    val_generated_maps = model.predict(val_meta.drop(columns=['datetime']).values)
    test_generated_maps = model.predict(test_meta.drop(columns=['datetime']).values)

    print('maps', train_generated_maps.shape)
    print('maps', test_generated_maps.shape)
    print('maps', val_generated_maps.shape)


    train_maps_errors = {}
    val_maps_errors = {}
    test_maps_errors = {}
    for error, func in errors.items():
        train_maps_errors[error] = mean_error_by_maps(train_generated_maps, train_maps, func)
        val_maps_errors[error] = mean_error_by_maps(val_generated_maps, val_maps, func)
        test_maps_errors[error] = mean_error_by_maps(test_generated_maps, test_maps, func)
    train_meta_with_errors = train_meta.assign(**train_maps_errors)
    val_meta_with_errors = val_meta.assign(**val_maps_errors)
    test_meta_with_errors = test_meta.assign(**test_maps_errors)

    mean_errors = {'index': ['train', 'val', 'test']}
    for error, func in errors.items():
        mean_errors[error] = []
        mean_errors[error].append(mean_error(train_generated_maps, train_maps, func))
        mean_errors[error].append(mean_error(val_generated_maps, val_maps, func))
        mean_errors[error].append(mean_error(test_generated_maps, test_maps, func))
    mean_errors = pd.DataFrame(mean_errors)

    print('errors', train_meta_with_errors.shape)
    print('errors', test_meta_with_errors.shape)
    print('errors', val_meta_with_errors.shape)

    if os.path.isdir('/tmp/errors'):
        os.system('rm -rf /tmp/errors')
    os.mkdir('/tmp/errors')
    train_meta_with_errors.to_csv('/tmp/errors/train_meta.csv', index=False)
    val_meta_with_errors.to_csv('/tmp/errors/val_meta.csv', index=False)
    test_meta_with_errors.to_csv('/tmp/errors/test_meta.csv', index=False)    
    mean_errors.to_csv('/tmp/errors/mean_errors.csv', index=False)

    mlflow.log_artifact('/tmp/errors')

    