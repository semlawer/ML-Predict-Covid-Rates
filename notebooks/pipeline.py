import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def _normalize_helper(df, cols, scaler=None):

    # Normalizing train set
    if(scaler is None):
      scaler = StandardScaler()
      normalized_features = pd.DataFrame(scaler.fit_transform(df[cols]))

    # Normalizing test set (with the values based on the training set)
    else:
      normalized_features = pd.DataFrame(scaler.transform(df[cols]))

    normalized_features.columns = cols
    normalized_df = pd.concat([df.drop(columns=cols).reset_index(drop=True), normalized_features], axis=1)

    return normalized_df, scaler

def normalize(train, test, cols):
    """
    Normalize training and test data
    """
    training_normalized, scaler = _normalize_helper(train, cols)
    test_normalized, _ = _normalize_helper(test, cols, scaler)

    return training_normalized, test_normalized


def one_hot_encode(df, var_list):
    """
    Helper function for standardize_encoding
    """
    for var in var_list:
        dummies = pd.get_dummies(df[var], prefix=var)
        df = pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

    return df.drop(columns=var_list)
