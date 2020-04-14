import os
import plotly.express as px
import pandas as pd
import json

MAIN_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(MAIN_DIR, 'upstream\\data')

def load_covid_data(source='usafacts'):
    yesterday = date.today() - timedelta(days=1)
    if source == 'usafacts':
        df = pd.read_csv(os.path.join(DATA_DIR, 'us\\covid\\confirmed_cases.csv'), 
            dtype={'countyFIPS':str})
        yesterday = f'{yesterday.month}/{yesterday.day}/{yesterday.year}'
        yesterday_cases = df.loc[:, ['State', 'countyFIPS', 'County Name', yesterday_col]]
    elif source == 'nytimes':
        raise NotImplementedError
    else:
        raise ValueError('Source not recognized. Options are: usafacts, nytimes')