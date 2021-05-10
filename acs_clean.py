import numpy as np
import pandas as pd
import os
import re


def read_and_clean_data(csv_file):
    '''
    Create pandas dataframe from csv file and extracting only columns with data 
    estimates and percentages

    Input: csv_file

    Retuns: df, a clean dataframe to extract desired columns from
    '''

    if not os.path.exists(csv_file):
        return None
    all_data = pd.read_csv(csv_file, low_memory=False).drop([0])
    df = all_data[all_data.columns.drop(all_data.filter(like='Margin of Error', axis=1))]
    cols = list(df.columns)
    for i, col in enumerate(cols):
        cols[i] = re.sub('Estimate!!', '', col)
    df.columns = cols
    
    return df

def summarize_demographics(csv_file):
    '''
     Extracts desired data from demographic data available from ACS surveys. 

    Inputs:
        csv_file: downloaded from ACS website

    Returns:
        df: panda dataframe with column data on age, race, and gender.
    '''

    build = ['geo_id', 'county_state','total_pop', 'male', 'perc_male','female', 'perc_female',
            'age_under5', 'p_age_under5', 'age_5_9', 'p_age_5_9', 'age_10_14', 'p_age_10_14',
            'age_15_19', 'p_age_15_19', 'age_20_24', 'p_age_20_24', 'age_25_34', 'p_age_25_34',
            'age_35_44', 'p_age_35_44', 'age_45_54', 'p_age_45_54', 'age_55_59', 'p_age_55_59',
            'age_60_64', 'p_age_60_64', 'age_65_74', 'p_age_65_74', 'age_75_84', 'p_age_75_84',
            'age_85over', 'p_age_85over', 'age_median', 'age_62over', 'p_age_62over', 'age_65over',
            'p_age_65over', "white", "p_white", "black", 'p_black', 'native', 'p_native', 'asian',
            'p_asian', 'hawaiian', 'p_hawaiian', 'other_race', 'p_other_race', 'hispanic', 'p_hispanic',
            'housing_units']

    extract = ["GEO_ID", "NAME", "DP05_0001E", "DP05_0002E", "DP05_0002PE", "DP05_0003E", "DP05_0003PE",
    "DP05_0005E", "DP05_0005PE", "DP05_0006E", "DP05_0006PE", "DP05_0007E","DP05_0007PE", "DP05_0008E",
    "DP05_0008PE", "DP05_0009E", "DP05_0009PE", "DP05_0010E", "DP05_0010PE", "DP05_0011E", "DP05_0011PE", 
    "DP05_0012E","DP05_0012PE", "DP05_0013E", "DP05_0013PE", "DP05_0014E", "DP05_0014PE", "DP05_0015E", 
    "DP05_0015PE", "DP05_0016E", "DP05_0016PE", "DP05_0017E", "DP05_0017PE", "DP05_0018E", 
    "DP05_0023E", "DP05_0023PE", "DP05_0024E", "DP05_0024PE", "DP05_0037E", "DP05_0037PE", "DP05_0038E",
    "DP05_0038PE", "DP05_0039E", "DP05_0039PE", "DP05_0044E", "DP05_0044PE", "DP05_0052E", "DP05_0052PE", 
    "DP05_0057E", "DP05_0057PE", "DP05_0071E", "DP05_0071PE", "DP05_0086E"]
    
    demo = read_and_clean_data(csv_file)
    demo = demo.loc[:, extract]
    demo.columns = build
    #demo.iloc[:, 1:] = demo.iloc[:, 1:].apply(pd.to_numeric)
    demo[['county','state']] = demo['county_state'].str.split(', ',expand=True)
    df = demo[demo["state"].isin(["Wisconsin", "Illinois", "North Dakota", "South Dakota", "Nebraska", 
    "Kansas", "Michigan", "Indiana", "Minnesota", "Iowa", "Missouri", "Ohio"])]
    #df["geo_id"] = df["geo_id"].astype("str")
    df["fips"] = df["geo_id"].str[-5:]
    print(df.dtypes)

    return df

def summarize_poverty(csv_file):
    '''
     Extracts desired data from poverty data available from ACS surveys. 

    Inputs:
        csv_file: downloaded from ACS website

    Returns:
        df: panda dataframe with column data on poverty level.
    '''

    build = ['geo_id', 'county_state','below_50_pov', 'below_125_pov', 'below_150_pov', 
            'below_185_pov', 'below_200_pov', 'below_300_pov', 'below_400_pov', 
            'below_500_pov', 'below_pov', 'male_below_pov', 'female_below_pov',]

    extract = ["GEO_ID", "NAME", "S1701_C01_038E", 'S1701_C01_039E', 'S1701_C01_040E', 'S1701_C01_041E', 
            'S1701_C01_042E', 'S1701_C01_043E', 'S1701_C01_044E', 'S1701_C01_045E', 'S1701_C02_001E',
            'S1701_C02_011E', 'S1701_C02_012E']
    
    poverty = read_and_clean_data(csv_file)
    poverty = poverty.loc[:, extract]
    poverty.columns = build
    poverty[['county','state']] = poverty['county_state'].str.split(', ',expand=True)
    df = poverty[poverty["state"].isin(["Wisconsin", "Illinois", "North Dakota", "South Dakota", "Nebraska", 
    "Kansas", "Michigan", "Indiana", "Minnesota", "Iowa", "Missouri", "Ohio"])]
    print(df.state.unique())
    df["fips"] = df["geo_id"].str[-5:]

    return df

def acs_full(demographics, poverty):
    '''
    Merge all the ACS data sets together
    '''
    acs = demographics.merge(poverty, on=["fips", "geo_id", "county_state", "state", "county"]
        , how="outer")
    acs["age_under14"] = acs["age_under5"] + acs['age_5_9'] + acs['age_10_14']
    acs["p_under14"] =  acs["p_age_under5"] + acs['p_age_5_9'] + acs['p_age_10_14']
    acs["non_white"] = pd.to_numeric(acs["total_pop"]) - pd.to_numeric(acs["white"])
    acs["p_non_white"] = 100 - pd.to_numeric(acs["p_white"])
    acs.drop(["age_under5", 'age_5_9', 'age_10_14', "p_age_under5", 'p_age_5_9', 'p_age_10_14', \
        'age_65_74', 'p_age_65_74', 'age_75_84', 'p_age_75_84', 'age_85over', 'p_age_85over',
        'geo_id', 'county_state'], axis = 1, inplace = True)
    
    return acs

def export_data(acs):
    '''
    Export data to csv to use in the build
    '''
    col = acs.pop("fips")
    acs.insert(0, col.name, col)
    acs.to_csv("Data/ACS Data.csv", index = False)