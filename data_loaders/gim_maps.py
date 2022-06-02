# Converts GIM maps to numpy array and meta file

import ionex
import numpy as np
import sys
import os 
import pandas as pd
import warnings 
from pathlib import Path
import warnings
import tqdm
warnings.filterwarnings("ignore")

DTYPE = [('lat', 'float'), ('lon', 'float'), ('vals', 'float')]

def ionex_to_map2d(filename):
    with open(filename) as file:
        try:
            for ionex_map in ionex.reader(file):
                data = []
                lats = ionex_map.grid.latitude
                lons = ionex_map.grid.longitude
                tec_iter = iter(ionex_map.tec)
                lon_grid = np.arange(lons.lon1, lons.lon2 + lons.dlon, lons.dlon)
                lat_grid = np.arange(lats.lat1, lats.lat2 + lats.dlat, lats.dlat)
                for lat in lat_grid:
                    for lon in lon_grid:
                        data.append((lat, lon, next(tec_iter)))
                yield np.array(data, dtype=DTYPE), ionex_map.epoch
        except ValueError as e:
            msg = f'For {filename} error {e} occur '
            msg += 'Probably file contains data for 24:00:00 '
            msg += ' which is the same as 00:00:00 of next day'
            msg += ' no data are loss in that case'
            warnings.warn(msg)

import matplotlib.pyplot as plt

if __name__ == '__main__':
    gim_path = Path(sys.argv[1])
    gim_files = [ gim_path / f for f in os.listdir(gim_path)]
    
    maps = []
    maps_by_time = {}
    for directory, directories, files in os.walk(gim_path):
        for f in tqdm.tqdm(files):
            data = ionex_to_map2d(os.path.join(directory, f))
            for gim, epoch in data:
                if epoch.minute == 0:
                    subtable = pd.DataFrame(gim)
                    subtable = subtable.assign(datetime=epoch)
                    subtable = subtable[subtable.datetime.dt.day == subtable.iloc[0].datetime.day]
                    subtable = subtable[subtable.lon != 180]
                    maps_by_time[epoch] = np.array(subtable.vals).reshape(71, 72)
    arr = []
    meta = pd.DataFrame({'datetime': sorted(maps_by_time.keys())})

    print('Assembling...')
    for date in tqdm.tqdm(meta.datetime):
        arr.append(maps_by_time[date])  
    arr = np.array(arr)
    np.save('maps', arr)
    meta.to_csv('meta.csv', index=False)
