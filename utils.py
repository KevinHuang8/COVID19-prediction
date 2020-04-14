import os
import datetime 
from datetime import date, timedelta
import pandas as pd
import numpy as np
import json

MAIN_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(MAIN_DIR, 'upstream\\data')

def load_covid_data(source='usafacts'):
    yesterday = date.today() - timedelta(days=2)
    if source == 'usafacts':
        df = pd.read_csv(os.path.join(DATA_DIR, 'us\\covid\\confirmed_cases.csv'), 
            dtype={'countyFIPS':str})
        df = df.rename(columns={'countyFIPS' : 'FIPS'})
        # Get cases from most recent day
        yesterday = f'{yesterday.month}/{yesterday.day}/{yesterday.strftime("%y")}'
        yesterday_cases = df.loc[:, ['FIPS', 'County Name', yesterday]]
        yesterday_cases = yesterday_cases.rename(columns={yesterday: 'cases'})
        # Add leading zero to FIPS codes with only 4 digits
        yesterday_cases.loc[yesterday_cases['FIPS'].str.len() == 4, 'FIPS'] = \
            '0' + yesterday_cases.loc[yesterday_cases['FIPS'].str.len() == 4, 'FIPS']
        # Remove state FIPS
        yesterday_cases = yesterday_cases[yesterday_cases['FIPS'].str.len() != 1]
        # Remove FIPS codes not in population data
        yesterday_cases = yesterday_cases[~yesterday_cases['FIPS'].isin(['02270', '06000'])]
        yesterday_cases = yesterday_cases.reset_index(drop=True)
        # Add log data
        logcases = np.log10(yesterday_cases['cases']).replace([np.inf, -np.inf], 0)
        yesterday_cases['log_cases'] = pd.Series(logcases, index=yesterday_cases.index)
        return yesterday_cases
    elif source == 'nytimes':
        raise NotImplementedError
    else:
        raise ValueError('Source not recognized. Options are: usafacts, nytimes')