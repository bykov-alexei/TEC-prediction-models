import math
import numpy as np
import pandas as pd
import joblib as jl
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap, maskoceans
from scipy.interpolate import griddata as gd

from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import matplotlib as mpl

plotdir = 'plots3/'
plot_model_names = False

def generate_ticks(min, max, number):
    step = (max - min) // number
    ticks = [math.ceil(min)]
    for i in range(1, number):
        ticks.append(int(min+i*step))
    ticks.append(int(max))
    return ticks

def plot_mercater_map(mapdata, minvalue = None, maxvalue=None):
    a = mapdata.reshape((72,71))
    lat_arr = np.linspace(-87.5, 87.5, 71)
    lon_arr = np.linspace(-175, 180, 72)
    coord_map = np.concatenate([[np.array([lat_arr[index[1]], lon_arr[index[0]], v])] for index, v in np.ndenumerate(a)]).astype(np.float32)
    dat = pd.DataFrame(coord_map,columns = ['lat', 'lon', 'tec'])

    lat_max = dat.lat.max()
    lat_min = dat.lat.min()
    lon_max = dat.lon.max()
    lon_min = dat.lon.min()
    lats = dat.lat
    lons = dat.lon
    if not minvalue:
        minvalue = dat.tec.min()
    if not maxvalue:
        maxvalue = dat.tec.max()

    numcols, numrows = lon_arr.shape[0]*2, lat_arr.shape[0]*2
    xi = np.linspace(np.min(lons), np.max(lons), numcols)
    yi = np.linspace(np.min(lats), np.max(lats), numrows)
    xi, yi = np.meshgrid(xi, yi)
    zi = gd((lons, lats), dat.tec, (xi, yi), method='cubic')

    m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80,\
                llcrnrlon=lon_min, urcrnrlon=lon_max, lat_ts=20, resolution='c')
    m.drawmapboundary(fill_color='w')
    m.drawcoastlines(color='black')

    parallels = np.array([-70, -50, -30, 0, 30, 50, 70])
    m.drawparallels(parallels, labels=[False, False, False, False], fontsize=10)
    meridians = np.array([-120, -60, 0, 60, 120])
    m.drawmeridians(meridians, labels=[False, False, False, False], fontsize=10)

    clevs = np.linspace(minvalue, maxvalue, 128)
    cf = m.contourf(xi, yi, zi, clevs, cmap=plt.cm.jet, extend='both', latlon=True)
    cbar = m.colorbar(cf, location='bottom')
    ticks = np.arange(minvalue, maxvalue).astype(int)
    cbar.set_ticks(ticks)

