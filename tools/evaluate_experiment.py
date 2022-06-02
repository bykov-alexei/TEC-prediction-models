import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings("ignore")

gim_path = '/home/machine/TEC-models/data/maps.npy'
gim_meta_path = '/home/machine/TEC-models/data/meta.csv'
indices_path = '/home/machine/TEC-models/data/indices.csv'

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
parser.add_argument('--title', dest='title', required=True)
parser.add_argument('--start-date', dest='start_date', required=True)
parser.add_argument('--end-date', dest='end_date', required=True)
args = parser.parse_args()

experiment_id = args.experiment
dates = [datetime.datetime.strptime(date, '%Y-%m-%dT%H') for date in args.date]

client = MlflowClient()
experiment = client.get_run(experiment_id)

