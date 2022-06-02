import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import argparse
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from plotters.functions import plot_mercater_map

parser = argparse.ArgumentParser()
parser.add_argument('--title', dest='title', required=True)
parser.add_argument('--experiment', dest='experiment', required=True)
args = parser.parse_args()

experiment_id = args.experiment
title = args.title

client = MlflowClient()
experiment = client.get_run(experiment_id)

artifact_folder = experiment.info.artifact_uri.replace('file://', '')
shift = int(experiment.data.params['shift'])

try:
    model_type = 'nn'
    model = mlflow.pyfunc.load_model(f'runs:/{experiment_id}/dense_model')
except:
    model_type = 'gimli'
    model = mlflow.pyfunc.load_model(f'runs:/{experiment_id}/gimli_model')

meta = pd.read_csv(os.path.join(artifact_folder, 'training_data', 'meta.csv'))
meta.datetime = pd.to_datetime(meta.datetime)

original_meta = pd.read_csv('data/meta.csv')
original_meta.datetime = pd.to_datetime(original_meta.datetime)
original_meta = original_meta.assign(map_index=np.arange(len(original_meta)) + shift)
original_maps = np.load('data/maps.npy')

print(meta[meta.datetime.dt.year == 2020].shape)
meta_pred = meta[meta.datetime.dt.year == 2020].drop(columns=['datetime']).values
y_pred = model.predict(meta_pred)
if model_type == 'gimli':
    y_pred = y_pred.reshape(-1, 71, 72)
y_true = original_maps[original_meta[original_meta.datetime.dt.year == 2020].map_index]
print(y_pred.shape)
print(y_true.shape)

error = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))

size = 6
plt.rcParams['figure.figsize'] = (size, size*0.9)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['figure.dpi'] = 300

props = dict(boxstyle='round', facecolor='white', alpha=1)

plot_mercater_map(error.T, 0, 15)
plt.tight_layout()
plt.title(str(title))
plt.savefig(f'{title}.png')
plt.clf()