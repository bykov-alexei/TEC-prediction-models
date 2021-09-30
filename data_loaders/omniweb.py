# Loads data from omniweb.gsfc.nasa.gov
from io import StringIO
import re
import requests
from datetime import datetime
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from config import config

indices = ['Kp', 'R', 'ap', 'f10.7', 'AE', 'AL', 'AU']
start_date = datetime(1998, 1, 1)
end_date = datetime(2021, 1, 30)

def load_table(start_date, end_date, indices):
    indices_vars = {
        'Kp': '38',
        'R': '39',
        'ap': '49',
        'f10.7': '50',
        'AE': '41',
        'AL': '52',
        'AU': '53',
    }

    url = 'https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi'

    data = [
    ('activity', 'ftp'),
    ('res', 'hour'),
    ('spacecraft', 'omni2'),
    ('scale', 'Linear'),
    ('ymin', ''),
    ('ymax', ''),
    ('charsize', ''),
    ('symsize', '0.5'),
    ('symbol', '0'),
    ('imagex', '640'),
    ('imagey', '480'),
    ]

    data.append(('start_date', start_date.strftime('%Y%m%d')))
    data.append(('end_date', end_date.strftime('%Y%m%d')))

    for index in indices:
        data.append(('vars', indices_vars[index]))

    response = requests.post(url, data=data)
    files = re.findall(r'<a.*href=\"(.*\.lst)\".*>(.*)</a>', response.text)
    link, filename = files[0]
    response = requests.get(link)
    text = response.text

    table = pd.read_csv(StringIO(text), delim_whitespace=True, header=None)
    table.columns = ['YEAR', 'DAY', 'UT'] + indices 

    if 'f10.7' in table:
        table.loc[table['f10.7'] == 999.9, 'f10.7'] = np.nan
    if 'AE' in table:
        table.loc[table['AE'] == 9999, 'AE'] = np.nan
    if 'AL' in table:
        table.loc[table['AL'] == 99999, 'AL'] = np.nan
    if 'AU' in table:
        table.loc[table['AU'] == 99999, 'AU'] = np.nan

    return table

def save_table(table):
    c = config['feature_storage']
    engine = create_engine(f"mysql://{c['user']}:{c['password']}@{c['host']}:5432/{c['database']}")
    table.to_sql('indices', con=engine, if_exists='replace')


table = load_table(start_date, end_date, indices)
save_table(table)
