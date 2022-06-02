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
from plotters import example

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', dest='experiment', required=True)
parser.add_argument('--date', dest='date', nargs='+', required=True)
args = parser.parse_args()

experiment_id = args.experiment
dates = [datetime.datetime.strptime(date, '%Y-%m-%dT%H') for date in args.date]

client = MlflowClient()
experiment = client.get_run(experiment_id)


artifact_folder = experiment.info.artifact_uri.replace('file://', '')
shift = int(experiment.data.params['shift'])

model = mlflow.pyfunc.load_model(f'runs:/{experiment_id}/dense_model')

meta = pd.read_csv(os.path.join(artifact_folder, 'training_data', 'meta.csv')).drop(columns=['map_index', 'hours'])
meta.datetime = pd.to_datetime(meta.datetime)

metas = []

for date in dates:
    row = meta.drop(columns=['datetime'])[meta.datetime == (date - datetime.timedelta(hours=shift))]
    if len(row) == 0:
        print('No data for date', date.isoformat())
        continue
    data = row.iloc[0].values
    metas.append(data)

metas = np.array(metas)
predictions = model.predict(metas)
for date, prediction in zip(dates, predictions):
    example.plot(experiment_id + '.png', str(date), prediction.T, 25)
