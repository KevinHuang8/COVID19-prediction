import os
import datetime 
from datetime import date, timedelta
import pandas as pd
import numpy as np
import censusdata as cd
import json

MAIN_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(MAIN_DIR, 'upstream\\data')
OTHER_DATA_DIR = os.path.join(MAIN_DIR, 'other data')

### FIPS changes: 51515 > 51019, 46113 > 46102, 2158 > 2270

def load_covid_data(source='usafacts'):
    yesterday = date.today() - timedelta(days=2)
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

        cols_to_use = yesterday_deaths.columns.difference(yesterday_cases.columns)
        cases_deaths = pd.merge(yesterday_cases, yesterday_deaths[cols_to_use], how='outer', 
            left_index=True, right_index=True)

        # Add leading zero to FIPS codes with only 4 digits
        cases_deaths.loc[cases_deaths['FIPS'].str.len() == 4, 'FIPS'] = \
            '0' + cases_deaths.loc[cases_deaths['FIPS'].str.len() == 4, 'FIPS']
        # Remove state FIPS
        cases_deaths = cases_deaths[cases_deaths['FIPS'].str.len() != 1]
        # Remove FIPS codes not in population data
        cases_deaths = cases_deaths[~cases_deaths['FIPS'].isin(['02158', '06000'])]
        cases_deaths = cases_deaths.reset_index(drop=True)
        # Add log data
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

def load_demographics_data():
    demographics = pd.read_csv(os.path.join(OTHER_DATA_DIR, 'county_demographics.csv'), dtype={'FIPS':str})
    demographics.drop(['NAME'], axis=1, inplace=True)
    return demographics

def generate_demographics_data():
    d = cd.download('acs5', 2018, cd.censusgeo([('county', '*')]),
                                       ['DP05_0018E', 'DP05_0037PE', 'DP05_0038PE', 'DP05_0071PE',],
                                       tabletype='profile')
    #https://api.census.gov/data/2018/acs/acs1/profile/groups/DP05.html
    d = d.rename(columns={'DP05_0018E': 'median_age', 'DP05_0037PE':'pop_white', 'DP05_0038PE':'pop_black','DP05_0071PE':'pop_hispanic'})
    d = d[['median_age', 'pop_white', 'pop_black', 'pop_hispanic']]
    cd.exportcsv(os.path.join(OTHER_DATA_DIR, 'county_demographics_temp.csv'), d)
    df = pd.read_csv(os.path.join(OTHER_DATA_DIR, 'county_demographics_temp.csv'), dtype={'state':str,'county':str})
    df['FIPS'] = df['state'] + df['county']
    df.drop(['state', 'county'], axis=1, inplace=True)
    df = df[['FIPS', 'NAME', 'median_age', 'pop_white', 'pop_black', 'pop_hispanic']]
    df.replace('02158', '02270', inplace=True)
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