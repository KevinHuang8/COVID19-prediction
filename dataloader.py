import os
import datetime as dt
from datetime import date, timedelta
import pandas as pd
import numpy as np
import censusdata as cd
import json
from pathlib import Path

MAIN_DIR = Path(__file__).parent.absolute()
DATA_DIR = MAIN_DIR / 'upstream' / 'data'
OTHER_DATA_DIR = MAIN_DIR / 'other data'

### FIPS changes: 51515 > 51019, 46113 > 46102, 2158 > 2270

def fix_county_FIPS(df):
    d = df.copy()
    # Add leading zero to FIPS codes with only 4 digits
    d.loc[d['FIPS'].str.len() == 4, 'FIPS'] = \
        '0' + d.loc[d['FIPS'].str.len() == 4, 'FIPS']
    # Remove state FIPS
    d = d[d['FIPS'].str.len() != 1]

    # Remove FIPS codes not in population data
    d = d[~d['FIPS'].isin(['02158', '06000'])]
    d.sort_values('FIPS', inplace=True)
    d.reset_index(drop=True, inplace=True)
    return d

def calibrate_timeseries(t, *s, cutoff=50):
    '''
    all positional arguments are arrays of shape (n_counties, n_timesteps)
    that represent time series

    Shifts each time series in s & t so that they begin at the same index
    as when the corresponding time series in t reaches the cutoff. For example,
    only take data after the day where deaths reaches 50, and sets that day as
    time = 0. Deaths would be passed in as t, and all the other series
    (for example cases) that you want to calibrate with deaths is passed in
    as s. The end of each time series becomes padded with nan.

    **Assumes that all values are non-decreasing with time, or
    else the order gets messed up for that row**
        - There are a few instances where cumulative deaths
          decreases in the data, which is impossible in real life
    '''
    row, col = np.indices(t.shape)
    # Index of first day when value > cutoff for each row
    calibrated_start = np.argmax(t > cutoff, axis=1)
    calibrated_start = np.expand_dims(calibrated_start, axis=1)
    # Rows that contain values all < cutoff
    to_remove = np.all(t <= cutoff, axis=1)

    # For each row, get all values after the calibrated
    # start date, and move them to the front
    mask = col >= calibrated_start
    flipped_mask = mask[:,::-1]

    calibrated = []
    for x in ((t,) + s):
        a = np.copy(x).astype(float)
        a[flipped_mask] = a[mask]
        # Everything not meeting cutoff is nan
        a[~flipped_mask] = np.nan
        a[to_remove] = np.nan
        calibrated.append(a)

    return calibrated

def smooth_timeseries(t, size=5):
    '''Smooth the function by taking a moving average of "size" time steps'''
    average_filter = np.full((size, ), 1 / size)

    t = np.pad(t, [(0, 0), (size // 2, size // 2)], mode='edge')
    return np.apply_along_axis(lambda r: np.convolve(r, average_filter, mode='valid'),
        axis=1, arr=t)

def load_covid_raw():
    df_cases = pd.read_csv(DATA_DIR / 'us' / 'covid' / 'confirmed_cases.csv',
        dtype={'countyFIPS':str})
    df_cases = df_cases.rename(columns={'countyFIPS' : 'FIPS'})
    df_deaths = pd.read_csv(DATA_DIR / 'us' / 'covid' / 'deaths.csv',
        dtype={'countyFIPS':str})
    df_deaths = df_deaths.rename(columns={'countyFIPS' : 'FIPS'})

    df_deaths = fix_county_FIPS(df_deaths)
    df_cases = fix_county_FIPS(df_cases)

    df_cases.drop(['County Name', 'State', 'stateFIPS'], axis=1, inplace=True)
    df_deaths.drop(['County Name', 'State', 'stateFIPS'], axis=1, inplace=True)

    return df_cases, df_deaths

def reload_nyt_data(windows):
    print('Reloading NYT data... May take a minute...')
    rawdata = load_covid_raw()
    rawcases = rawdata[0]
    rawdeaths = rawdata[1]

    dat = pd.read_csv(DATA_DIR / "us" / "covid" / "nyt_us_counties.csv",
        parse_dates=[0],
        dtype={'fips':str})
    dat.loc[dat['county'] == 'New York City', 'fips'] = '36061'
    dat.loc[dat['state'] == 'Guam', 'fips'] = '66010'
    if windows:
        dat['date'] = dat['date'].dt.strftime('%#m/%#d/%y')
    else:
        dat['date'] = dat['date'].dt.strftime('%-m/%-d/%y')
    dat = dat.astype({'date' : str})
    data_cases = pd.DataFrame()
    data_deaths = pd.DataFrame()

    curr = dt.datetime.strptime('1/21/2020', '%m/%d/%Y')
    last = dt.datetime.strptime(dat.iloc[-1].date, '%m/%d/%y')
    data_cases['FIPS'] = np.nan
    while curr != last:
        curr = curr + timedelta(days=1)
        if windows:
            data_cases[curr.strftime('%#m/%#d/%y')] = np.nan
        else:
            data_cases[curr.strftime('%-m/%-d/%y')] = np.nan

    curr = dt.datetime.strptime('1/21/2020', '%m/%d/%Y')
    last = dt.datetime.strptime(dat.iloc[-1].date, '%m/%d/%y')
    data_deaths['FIPS'] = np.nan
    while curr != last:
        curr = curr + timedelta(days=1)
        if windows:
            data_deaths[curr.strftime('%#m/%#d/%y')] = np.nan
        else:
            data_deaths[curr.strftime('%-m/%-d/%y')] = np.nan

    NYT_fips = dat['fips'].unique()
    for index, row in rawcases.iterrows():
        fips = row['FIPS']
        if fips not in NYT_fips:
            data_cases = data_cases.append(row, ignore_index=True)
            continue
        r = dat[dat['fips'] == fips].drop(['fips', 'county', 'state',
            'deaths'], axis=1).T
        r.columns = r.iloc[0]
        r.drop('date', axis=0, inplace=True)
        r['FIPS'] = fips
        data_cases = data_cases.append(r, ignore_index=True, sort=False)
        #print('cases: ' + str(index))

    for index, row in rawdeaths.iterrows():
        fips = row['FIPS']
        if fips not in NYT_fips:
            data_deaths = data_deaths.append(row, ignore_index=True)
            continue
        r = dat[dat['fips'] == fips].drop(['fips', 'county', 'state',
            'cases'], axis=1).T
        r.columns = r.iloc[0]
        r.drop('date', axis=0, inplace=True)
        r['FIPS'] = fips
        data_deaths = data_deaths.append(r, ignore_index=True, sort=False)
        #print('deaths: ' + str(index))

    r = dat[dat['fips'] == '66010'].drop(['fips', 'county', 'state',
        'deaths'], axis=1).T
    r.columns = r.iloc[0]
    r.drop('date', axis=0, inplace=True)
    r['FIPS'] = '66010'
    data_cases = data_cases.append(r, ignore_index=True, sort=False)

    r = dat[dat['fips'] == '66010'].drop(['fips', 'county', 'state',
        'cases'], axis=1).T
    r.columns = r.iloc[0]
    r.drop('date', axis=0, inplace=True)
    r['FIPS'] = '66010'
    data_deaths = data_deaths.append(r, ignore_index=True, sort=False)

    data_deaths.drop('1/21/20', axis=1, inplace=True)
    data_deaths.fillna(0, inplace=True)
    data_deaths.to_csv(OTHER_DATA_DIR / 'nyt_deaths.csv')
    data_cases.drop('1/21/20', axis=1, inplace=True)
    data_cases.fillna(0, inplace=True)
    data_cases.to_csv(OTHER_DATA_DIR / 'nyt_cases.csv')

def load_covid_timeseries(source='nytimes', smoothing=5, cases_cutoff=200, log=False,
    deaths_cutoff=50, interval_change=1, reload_data=False, force_no_reload=False,
    windows=True):
    if source == 'nytimes':
        if not reload_data and not force_no_reload:
            df_cases = pd.read_csv(OTHER_DATA_DIR / 'nyt_cases.csv', dtype={'FIPS':str})
            nyt_raw = pd.read_csv(DATA_DIR / "us" / "covid" / "nyt_us_counties.csv"
                , dtype={'countyFIPS':str})
            last_date_available = dt.datetime.strptime(nyt_raw.iloc[-1].date,
                '%Y-%m-%d')
            last_date_checked =  dt.datetime.strptime(df_cases.columns[-1],
                '%m/%d/%y')
            if last_date_checked != last_date_available:
                reload_data = True

        if reload_data:
            reload_nyt_data(windows)

        df_cases = pd.read_csv(OTHER_DATA_DIR / 'nyt_cases.csv', dtype={'FIPS':str})
        df_deaths = pd.read_csv(OTHER_DATA_DIR / 'nyt_deaths.csv', dtype={'FIPS':str})

        df_deaths = df_deaths.iloc[:, 2:]
        df_cases = df_cases.iloc[:, 2:]

    elif source =='usafacts':
        df_cases = pd.read_csv(DATA_DIR / 'us' / 'covid' / 'confirmed_cases.csv',
            dtype={'countyFIPS':str})
        df_cases = df_cases.rename(columns={'countyFIPS' : 'FIPS'})
        df_deaths = pd.read_csv(DATA_DIR / 'us' / 'covid' / 'deaths.csv',
            dtype={'countyFIPS':str})
        df_deaths = df_deaths.rename(columns={'countyFIPS' : 'FIPS'})

        df_deaths = fix_county_FIPS(df_deaths)
        df_cases = fix_county_FIPS(df_cases)

        # Get rid of every column except for time series
        df_deaths = df_deaths.iloc[:, 4:]
        df_cases = df_cases.iloc[:, 4:]
    else:
        raise ValueError('Invalid Source. Must be "nytimes" or "usafacts".')

    if log:
        df_cases = np.log10(df_cases).replace([np.inf, -np.inf], 0)
        df_deaths = np.log10(df_deaths).replace([np.inf, -np.inf], 0)

    # Calibrate cases based on a cases cutoff
    cases, deaths = calibrate_timeseries(df_cases.values,
        df_deaths.values, cutoff=cases_cutoff)

    # This below does deaths calibration independently
    # deaths = calibrate_timeseries(df_deaths.values, cutoff=deaths_cutoff)

    # Get percentage change between 'interval_change' days
    with np.errstate(divide='ignore', invalid='ignore'):
        d = deaths[:, ::interval_change]
        c = cases[:, ::interval_change]
        deaths_pchange = np.diff(d) / d[:, :-1]
        cases_pchange = np.diff(c) / c[:, :-1]
    # all invalid percentage changes should be set to nan
    deaths_pchange = np.nan_to_num(deaths_pchange, nan=np.nan, posinf=np.nan, neginf=np.nan)
    cases_pchange = np.nan_to_num(cases_pchange, nan=np.nan, posinf=np.nan, neginf=np.nan)

    d_smoothed = smooth_timeseries(deaths_pchange, smoothing)
    c_smoothed = smooth_timeseries(cases_pchange, smoothing)

    cases_calib_smooth = smooth_timeseries(cases, smoothing)
    deaths_calib_smooth = smooth_timeseries(deaths, smoothing)

    return {'deaths_pc' : deaths_pchange,
            'deaths_pc_smoothed' : d_smoothed,
            'deaths_calibrated' : deaths,
            'deaths_raw' : df_deaths.values.astype(float),
            'deaths_calibrated_smoothed' : deaths_calib_smooth,
            'cases_pc' : cases_pchange,
            'cases_pc_smoothed' : c_smoothed,
            'cases_calibrated' : cases,
            'cases_raw' : df_cases.values.astype(float),
            'cases_calibrated_smoothed' : cases_calib_smooth}

def load_covid_static(source='usafacts', days_ago=2):
    yesterday = date.today() - timedelta(days=days_ago)
    if source == 'usafacts':
        df_cases = pd.read_csv(os.path.join(DATA_DIR, 'us\\covid\\confirmed_cases.csv'),
            dtype={'countyFIPS':str})
        df_cases = df_cases.rename(columns={'countyFIPS' : 'FIPS'})
        df_deaths = pd.read_csv(os.path.join(DATA_DIR, 'us\\covid\\deaths.csv'),
            dtype={'countyFIPS':str})
        df_deaths = df_deaths.rename(columns={'countyFIPS' : 'FIPS'})

        # Get data from most recent day
        yesterday = f'{yesterday.month}/{yesterday.day}/{yesterday.strftime("%y")}'
        yesterday_cases = df_cases.loc[:, ['FIPS', 'County Name', yesterday]]
        yesterday_deaths = df_deaths.loc[:, ['FIPS', 'County Name', yesterday]]
        yesterday_cases = yesterday_cases.rename(columns={yesterday: 'cases'})
        yesterday_deaths = yesterday_deaths.rename(columns={yesterday: 'deaths'})

        # Combine cases and deaths into one table for easy access
        cols_to_use = yesterday_deaths.columns.difference(yesterday_cases.columns)
        cases_deaths = pd.merge(yesterday_cases, yesterday_deaths[cols_to_use], how='outer',
            left_index=True, right_index=True)

        cases_deaths = fix_county_FIPS(cases_deaths)

        # Add log data for better graphing
        logcases = np.log10(cases_deaths['cases']).replace([np.inf, -np.inf], 0)
        logdeaths = np.log10(cases_deaths['deaths']).replace([np.inf, -np.inf], 0)
        cases_deaths['log_cases'] = pd.Series(logcases, index=cases_deaths.index)
        cases_deaths['log_deaths'] = pd.Series(logdeaths, index=cases_deaths.index)
        cases_deaths.sort_values('FIPS', inplace=True)
        cases_deaths = cases_deaths.reset_index(drop=True)
        return cases_deaths
    elif source == 'nytimes':
        raise NotImplementedError
    else:
        raise ValueError('Source not recognized. Options are: usafacts, nytimes')

def load_demographics_data(include_guam=True):
    demographics = pd.read_csv(os.path.join(OTHER_DATA_DIR, 'county_demographics.csv'), dtype={'FIPS':str})
    demographics.drop(['NAME'], axis=1, inplace=True)
    if not include_guam:
        demographics = demographics.iloc[:-1]
    demographics['pop_density'] = demographics['total_pop'] / demographics['area']
    demographics['p60_plus'] = demographics['60plus'] / demographics['total_pop']
    return demographics

def generate_demographics_data():
    '''Don't really need to call this again after the data is already generated, since demographics data
    doesn't change. Except if you want to change the demographics data.'''
    d = cd.download('acs5', 2018, cd.censusgeo([('county', '*')]),
                                       ['DP05_0018E', 'DP05_0037PE', 'DP05_0038PE', 'DP05_0071PE',],
                                       tabletype='profile')
    #Find variable names for data you want here:
    #https://api.census.gov/data/2018/acs/acs1/profile/groups/DP05.html
    d = d.rename(columns={'DP05_0018E': 'median_age', 'DP05_0037PE':'pop_white', 'DP05_0038PE':'pop_black','DP05_0071PE':'pop_hispanic'})
    d = d[['median_age', 'pop_white', 'pop_black', 'pop_hispanic']]
    cd.exportcsv(os.path.join(OTHER_DATA_DIR, 'county_demographics_temp.csv'), d)
    df = pd.read_csv(os.path.join(OTHER_DATA_DIR, 'county_demographics_temp.csv'), dtype={'state':str,'county':str})
    df['FIPS'] = df['state'] + df['county']
    df.drop(['state', 'county'], axis=1, inplace=True)
    df = df[['FIPS', 'NAME', 'median_age', 'pop_white', 'pop_black', 'pop_hispanic']]
    df.replace('02158', '02270', inplace=True)
    # Remove puerto rico
    df = df.drop(df[df['FIPS'].str[:2] == '72'].index)
    df.sort_values('FIPS', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Get population data
    population = pd.read_csv(os.path.join(DATA_DIR, 'us\\demographics\\county_populations.csv'), dtype={'FIPS':str})
    population.loc[population['FIPS'].str.len() == 4, 'FIPS'] = '0' + population.loc[population['FIPS'].str.len() == 4, 'FIPS']
    population.replace('02158', '02270', inplace=True)
    population.sort_values('FIPS', inplace=True)
    population.reset_index(drop=True, inplace=True)

    # Get land area data
    land = pd.read_csv(os.path.join(DATA_DIR, 'us\\demographics\\county_land_areas.csv'), dtype={'County FIPS':str}, engine='python')
    land.rename(columns={'County FIPS': 'FIPS'}, inplace=True)
    land = land.loc[:, ['FIPS', 'Area in square miles - Land area']]
    land.rename(columns={'Area in square miles - Land area' : 'area'}, inplace=True)
    land.loc[land['FIPS'].str.len() == 4, 'FIPS'] = '0' + land.loc[land['FIPS'].str.len() == 4, 'FIPS']
    land.replace('46113', '46102', inplace=True)
    land.drop(land[land['FIPS'] == '51515'].index, inplace=True)
    # Remove Puerto Rican data
    land.drop(land[land['FIPS'].str[:2] == '72'].index, inplace=True)
    land.sort_values('FIPS', inplace=True)
    land.reset_index(drop=True, inplace=True)

    demographics = population.merge(land)
    demographics = pd.merge(demographics, df, on=['FIPS'], how='outer')

    demographics.to_csv(os.path.join(OTHER_DATA_DIR, 'county_demographics.csv'), index=False, sep=',')

def generate_demographics_data2(include_age_breakdown=False):
    '''Use the other one. Here for reference'''

    # Get population data
    population = pd.read_csv(os.path.join(DATA_DIR, 'us\\demographics\\county_populations.csv'), dtype={'FIPS':str})
    population.loc[population['FIPS'].str.len() == 4, 'FIPS'] = '0' + population.loc[population['FIPS'].str.len() == 4, 'FIPS']
    population.replace('02158', '02270', inplace=True)
    population.sort_values('FIPS', inplace=True)
    population.reset_index(drop=True, inplace=True)

    # Get land area data
    land = pd.read_csv(os.path.join(DATA_DIR, 'us\\demographics\\county_land_areas.csv'), dtype={'County FIPS':str}, engine='python')
    land.rename(columns={'County FIPS': 'FIPS'}, inplace=True)
    land = land.loc[:, ['FIPS', 'Area in square miles - Land area']]
    land.rename(columns={'Area in square miles - Land area' : 'area'}, inplace=True)
    land.loc[land['FIPS'].str.len() == 4, 'FIPS'] = '0' + land.loc[land['FIPS'].str.len() == 4, 'FIPS']
    land.replace('46113', '46102', inplace=True)
    land.drop(land[land['FIPS'] == '51515'].index, inplace=True)
    # Remove Puerto Rican data
    land.drop(land[land['FIPS'].str[:2] == '72'].index, inplace=True)
    land.sort_values('FIPS', inplace=True)
    land.reset_index(drop=True, inplace=True)

    # Get age/gender/race data (note, many counties missing)
    df = pd.read_csv(os.path.join(DATA_DIR, 'us\\demographics\\acs_2018.csv'), dtype={'FIPS':str}, engine='python')
    df.loc[df['FIPS'].str.len() == 4, 'FIPS'] = '0' + df.loc[df['FIPS'].str.len() == 4, 'FIPS']
    df.sort_values('FIPS', inplace=True)
    df.rename(columns={'Estimate!!SEX AND AGE!!Total population!!Sex ratio (males per 100 females)':'mf_ratio',
                       'Estimate!!SEX AND AGE!!Total population!!Median age (years)' : 'median_age',
                       'Percent Estimate!!SEX AND AGE!!Total population!!Under 5 years' : 'pop_under5',
                      'Percent Estimate!!SEX AND AGE!!Total population!!5 to 9 years' : 'pop_5to9',
                      'Percent Estimate!!SEX AND AGE!!Total population!!10 to 14 years' : 'pop_10to14',
                      'Percent Estimate!!SEX AND AGE!!Total population!!15 to 19 years' : 'pop_15to19',
                      'Percent Estimate!!SEX AND AGE!!Total population!!20 to 24 years' : 'pop_20to24',
                      'Percent Estimate!!SEX AND AGE!!Total population!!25 to 34 years' : 'pop_25to34',
                      'Percent Estimate!!SEX AND AGE!!Total population!!35 to 44 years' : 'pop_35to44',
                      'Percent Estimate!!SEX AND AGE!!Total population!!45 to 54 years' : 'pop_45to54',
                      'Percent Estimate!!SEX AND AGE!!Total population!!55 to 59 years' : 'pop_55to59',
                      'Percent Estimate!!SEX AND AGE!!Total population!!60 to 64 years' : 'pop_60to64',
                      'Percent Estimate!!SEX AND AGE!!Total population!!65 to 74 years' : 'pop_65to74',
                      'Percent Estimate!!SEX AND AGE!!Total population!!75 to 84 years' : 'pop_75to84',
                      'Percent Estimate!!SEX AND AGE!!Total population!!85 years and over' : 'pop_over85',
                      'Percent Estimate!!RACE!!Total population!!One race!!White' : 'pop_white',
                      'Percent Estimate!!RACE!!Total population!!One race!!Black or African American' : 'pop_black',
                      'Percent Estimate!!HISPANIC OR LATINO AND RACE!!Total population!!Hispanic or Latino (of any race)' : 'pop_hispanic'}, inplace=True)
    if include_age_breakdown:
        df = df.loc[:, ['FIPS', 'mf_ratio', 'median_age', 'pop_under5', 'pop_5to9', 'pop_15to19', 'pop_20to24', 'pop_25to34', 'pop_35to44', 'pop_45to54',
                        'pop_55to59', 'pop_60to64', 'pop_65to74', 'pop_75to84', 'pop_over85', 'pop_white', 'pop_black', 'pop_hispanic']]
    else:
        df = df.loc[:, ['FIPS', 'mf_ratio', 'median_age', 'pop_white', 'pop_black', 'pop_hispanic']]
    df = df.drop(df[df['FIPS'].str[:2] == '72'].index)

    demographics = population.merge(land)
    demographics = pd.merge(demographics, df, on=['FIPS'], how='outer')
    demographics[['pop_white', 'pop_black', 'pop_hispanic']] = demographics[['pop_white', 'pop_black', 'pop_hispanic']].apply(pd.to_numeric, errors='coerce')

    d = demographics.copy()

    # Fill in NaN values from acs data
    # age breakdown and mf_ratio are taken as national average, while race is taken as state average
    # more advanced/accurate technique would be to replace NaNs by average of similar counties, found through clustering
    statedemo = pd.read_csv('../other data/state_demographics.csv', dtype={'stateFIPS':str})
    statedemo.loc[:, ['White', 'Black', 'Hispanic']] *= 100

    if include_age_breakdown:
        demographics[['mf_ratio', 'median_age', 'pop_under5', 'pop_5to9', 'pop_15to19', 'pop_20to24', 'pop_25to34', 'pop_35to44', 'pop_45to54',
                        'pop_55to59', 'pop_60to64', 'pop_65to74', 'pop_75to84', 'pop_over85']] = \
        demographics[['mf_ratio', 'median_age', 'pop_under5', 'pop_5to9', 'pop_15to19', 'pop_20to24', 'pop_25to34', 'pop_35to44', 'pop_45to54',
                        'pop_55to59', 'pop_60to64', 'pop_65to74', 'pop_75to84', 'pop_over85']].fillna(value=demographics.mean().round(1))
    else:
        demographics[['mf_ratio', 'median_age']] = demographics[['mf_ratio', 'median_age']].fillna(value=demographics.mean().round(1))

    demographics['stateFIPS'] = demographics['FIPS'].str[:2]
    t = pd.merge(demographics, statedemo, on=['stateFIPS'])
    demographics['pop_white'].fillna(t['White'], inplace=True)
    demographics['pop_hispanic'].fillna(t['Hispanic'], inplace=True)
    demographics['pop_black'].fillna(t['Black'], inplace=True)
    del demographics['stateFIPS']

    d.to_csv(os.path.join(OTHER_DATA_DIR, 'demographics_raw.csv'), sep=',')
    demographics.to_csv(os.path.join(OTHER_DATA_DIR, 'demographics.csv'), sep=',')
