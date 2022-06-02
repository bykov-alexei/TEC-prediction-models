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
parser.add_argument('--date', dest='date', type=str, required=True)
args = parser.parse_args()

experiment_id = args.experiment
date = datetime.datetime.strptime(args.date, '%Y-%m-%d')

client = MlflowClient()
experiment = client.get_run(experiment_id)


artifact_folder = experiment.info.artifact_uri.replace('file://', '')
shift = int(experiment.data.params['shift'])

model = mlflow.pyfunc.load_model(f'runs:/{experiment_id}/dense_model')

meta = pd.read_csv(os.path.join(artifact_folder, 'training_data', 'meta.csv'))
meta.datetime = pd.to_datetime(meta.datetime)

original_meta = pd.read_csv('data/meta.csv')
original_meta.datetime = pd.to_datetime(original_meta.datetime)
original_maps = np.load('data/maps.npy')

metas = []
maps = []
for hour in range(25):
    timestamp = date + datetime.timedelta(hours=hour)
    row = meta.drop(columns=['datetime'])[meta.datetime == (timestamp - datetime.timedelta(hours=shift))]
    row2 = original_maps[original_meta.datetime == timestamp]
    # print(row, row2)
    if len(row) == 0 or len(row2) == 0:
        print('No data for date', date.isoformat())
        exit(0)
    data = row.iloc[0].values
    metas.append(data)
    maps.append(row2[0])

metas = np.array(metas)
predictions = model.predict(metas)

text = "#NN-prediction\n# Year, nday, UT, Lon, Lat, TEC(TECU)\n"
text2 = "#Actual map\n# Year, nday, UT, Lon, Lat, TEC(TECU)\n"

print(np.sqrt(np.mean((maps[0] - predictions[0]) ** 2)))

# plt.imsave(f'{date.year}-{date.month}-{date.day}-nn-prediction.png', maps[0])
# plt.imsave(f'{date.year}-{date.month}-{date.day}-actual-map.png', predictions[0])

for hour, prediction, map in zip(range(25), predictions, maps):
    for i in range(71):
        for j in range(72):
            lon = -175 + j * 5
            lat = 87.5 - i * 2.5

            value = prediction[i, j]
            value2 = map[i, j]
            line1 = "%4d %3d %4.1f %6.1f %6.1f %6.2f\n" % (date.year, date.timetuple().tm_yday, hour, lon, lat, value)
            line2 = "%4d %3d %4.1f %6.1f %6.1f %6.2f\n" % (date.year, date.timetuple().tm_yday, hour, lon, lat, value2)

            text += line1
            text2 += line2

with open(f'{date.year}-{date.month}-{date.day}-nn-prediction.txt', 'w') as f:
    f.write(text)

with open(f'{date.year}-{date.month}-{date.day}-actual-map.txt', 'w') as f:
    f.write(text2)
