import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# TO REPLACE
def normalize(train_data, test_data, features):
    '''
    takes 2 datasets (testing data and trainning data) and a list of features to normalize 
    and changes columns to normalized new columns.
    '''
    
    scaler = StandardScaler()
    train = train_data.copy()
    test = test_data.copy()
    
    normal_train = scaler.fit_transform(train[features])
    normal_test = scaler.transform(test[features])
    
    for i, col in enumerate(features):
        train.loc[:, col] = normal_train[:, i]
        test.loc[:, col] = normal_test[:, i]

    return train, test


# BL: my one hot encoder function from the last assignment
def one_hot_encode(df, var_list):
    """
    Helper function for standardize_encoding
    """
    for var in var_list:
        dummies = pd.get_dummies(df[var], prefix=var)
        df = pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

    return df.drop(columns=var_list)
