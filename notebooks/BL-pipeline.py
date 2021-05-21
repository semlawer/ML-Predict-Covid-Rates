import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import datetime
from sklearn.inspection import permutation_importance

# 1. Read Data
"""
Read in csv
"""
def read_data(csv):
    return pd.read_csv(csv)

# 2. Explore Data

"""
Summarize continuous variables by group
"""
def summarize_cont_by_group(df, group_vars, cont_vars):
    df = df[cont_vars+group_vars]
    return df.groupby(group_vars).agg(['mean'])

"""
Plot distribution of categorical variables
"""
def plot_cat_distribution(df, var, sort=False):
    sns.set(rc={'figure.figsize':(11, 4)})

    df_plt = pd.DataFrame(df[var].value_counts())\
            .reset_index()\
            .rename(columns={var: "Count", "index": var})

    if sort:
        df_plt = df_plt.sort_values(by=['Count'], ascending=False).reset_index()

    sns.barplot(data=df_plt, x=df_plt.index, y="Count")
    plt.xticks(np.arange(len(df_plt)), rotation = 90, labels=df_plt[var].values)
    plt.xlabel(var)
    plt.title("Count by {}".format(var))

# 3. Create Training and Test Set
"""
Create training and test set
"""
def create_train_test(df, test_size=0.2):
    train, test = train_test_split(df, test_size=test_size, random_state=30254)
    print("size of train: {}".format(train.shape[0]))
    print("size of test: {}".format(test.shape[0]))

    return train, test

# 4. Pre-process Data
"""
Convert boolean field to numeric
"""
def bool_to_numeric(test, train, var, type=int):
    out = []
    for df in [test, train]:
        out.append(df.astype({var: int}))
    return tuple(out)

"""
Impute missings with mean or median
"""
def impute_missings(dfs, var_list, stat):
    out = []

    for df in dfs:
        for var in var_list:
            if stat=="mean":
                fill_val = df[var].mean()
            else:
                fill_val = df[var].median()

            df[var] = df[var].fillna(fill_val)

        out.append(df)

    return tuple(out)

"""
Helper function for normalize_train_test
"""
def normalize(df, cols, scaler=None, append=False):

    # Normalizing train set
    if(scaler is None):
      scaler = StandardScaler()
      normalized_features = pd.DataFrame(scaler.fit_transform(df[cols]))

    # Normalizing test set (with the values based on the training set)
    else:
      normalized_features = pd.DataFrame(scaler.transform(df[cols]))

    if append:
        # Add _NORM suffix to normalized columns
        normalized_features.columns = [colname + "_NORM" for colname in cols]

        # Column bind normalized columns onto original df
        normalized_df = pd.concat([df.reset_index(drop=True), normalized_features], axis = 1)
    else:
        normalized_features.columns = cols
        normalized_df = pd.concat([df.drop(columns=cols).reset_index(drop=True), normalized_features], axis=1)

    return normalized_df, scaler

"""
Normalize training and test data
"""
def normalize_train_test(train, test, cols, append=False):
    training_normalized, scaler = normalize(train, cols, append=append)
    test_normalized, _ = normalize(test, cols, scaler, append=append)

    return training_normalized, test_normalized

"""
Check range of numerical variable
"""
def check_range(df, var_list, rg_min=None, rg_max=None):
    for var in var_list:
        if rg_min!=None:
            if len(df[df[var]<rg_min])>0:
                print("{} has value below allowed minimum".format(var))
                print(df[var][df[var]<rg_min].head())

        if rg_max!=None:
            if len(df[df[var]>rg_max])>0:
                print("{} has value above allowed maximum".format(var))
                print(df[var][df[var]>rg_max].head())

    return "Ranges checked"

"""
Helper function for standardize_encoding
"""
def one_hot_encode(df, var_list):
    for var in var_list:
        dummies = pd.get_dummies(df[var], prefix=var)
        df = pd.concat([df.reset_index(drop=True), dummies], axis=1)

    return df.drop(columns=var_list)

"""
One-hot encode variables and standardize across training and test
"""
def standardize_encoding(train, test, var_list):
    train = one_hot_encode(train, var_list)
    test = one_hot_encode(test, var_list)

    only_in_train = np.setdiff1d(train.columns,test.columns)
    only_in_test = np.setdiff1d(test.columns,train.columns)

    for col in only_in_train:
        test_norm[col]=0

    for col in only_in_test:
        train_norm[col]=0

    return train, test

"""
Perform grid search
"""
def grid_search(MODELS, GRID, train_norm, test_norm, yvar):
    # Begin timer
    start = datetime.datetime.now()

    # Initialize results data frame
    # YOUR CODE HERE
    results = pd.DataFrame(columns = ["Model", "Parameters", "Training Accuracy", "Testing Accuracy"])
    feat_import = pd.DataFrame()

    model_num = 0

    # Loop over models
    for model_key in MODELS.keys():

        # Loop over parameters
        for params in GRID[model_key]:
            print("Training model:", model_key, "|", params)

            # Create model
            model = MODELS[model_key]
            model.set_params(**params)

            # Fit model on training set
            # YOUR CODE HERE
            model.fit(train_norm.drop(columns=[yvar]), train_norm[yvar].values.ravel())

            # Predict on testing set
            # YOUR CODE HERE
            model.predict(pd.DataFrame(test_norm.drop(columns=[yvar])))

            # Evaluate predictions
            # YOUR CODE HERE
            train_acc = model.score(train_norm.drop(columns=[yvar]), train_norm[yvar].values.ravel())
            test_acc = model.score(test_norm.drop(columns=[yvar]), test_norm[yvar].values.ravel())

            # Feature importance
            if model_key=="GaussianNB":
                feat_import[model_num] = pd.Series(permutation_importance(model, test_norm.drop(columns=[yvar]), test_norm[yvar].values.ravel()))
            else:
                feat_import[model_num] = pd.Series(model.coef_[0])

            # Store results in your results data frame
            # YOUR CODE HERE
            results = results.append({"Model": model_key,
                            "Parameters": params,
                            "Training Accuracy": train_acc,
                            "Testing Accuracy": test_acc},
                            ignore_index = True)

            model_num += 1

    # End timer
    stop = datetime.datetime.now()
    print("Time Elapsed:", stop - start)

    return results, feat_import