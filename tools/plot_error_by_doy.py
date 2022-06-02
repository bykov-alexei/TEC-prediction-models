import os
import numpy as np
import argparse
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

parser = argparse.ArgumentParser()
parser.add_argument('--experiments', dest='experiments', nargs='+', required=True)
parser.add_argument('--titles', dest='titles', nargs='+', required=True)

args = parser.parse_args()

inch_width = 17.4 / 2.54
plt.rcParams['figure.figsize'] = (inch_width, inch_width*3/4)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 300

experiments = args.experiments
titles = args.titles
assert len(experiments) == len(titles)

original_meta = pd.read_csv('data/indices.csv')
# original_meta.datetime = pd.to_datetime(original_meta.datetime)

client = MlflowClient()

for experiment, title in zip(experiments, titles):
    experiment = client.get_run(experiment)

    artifact_folder = experiment.info.artifact_uri.replace('file://', '')

    errors = pd.read_csv(os.path.join(artifact_folder, 'errors', 'val_meta.csv'))
    errors.datetime = pd.to_datetime(errors.datetime)
    tmp = errors[['datetime', 'rmse']]
    tmp.datetime = tmp.datetime.dt.dayofyear
    tmp = tmp.groupby(tmp['datetime'])['rmse'].mean()
    plt.plot(tmp.index, tmp.values, label=title)

tmp = (errors['datetime'].dt.dayofyear).value_counts()
plt.bar(tmp.index, tmp.values / sum(tmp.values) * 100, alpha=0.5, color='blue')

plt.xlabel('Day of Year')
plt.ylabel('RMSE, TECu')
plt.legend()
plt.savefig('error_by_doy.png')
