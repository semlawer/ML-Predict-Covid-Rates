# Mobility Data from Google

import pandas as pd
import numpy as np

def read_in(file):
    df = pd.read_csv(file)
    df.drop(["metro_area", "place_id", "country_region", "iso_3166_2_code"], inplace = True, axis = 1)
    df.columns = ["country", "state", "county", "fips code", "date", \
    "retail_rec", "grocery_pharm", "parks", "transit", "workplace", "residential"]
    df_IL = df[df["state"]== "Illinois"]
    return df_IL