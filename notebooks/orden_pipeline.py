import pandas as pd
import os
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import datetime


def read_data(file_path):
    '''
    This function checks if the given csv file path exists. If it does, reads
    in a given csv file into a Pandas dataframe.

    Inputs: file_path (str) - a path to the csv file to be read
    Returns: df (DataFrame) - the data as a dataframe
    '''
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    print("Please provide a valid file path.")


def explore_data_shape(df):
    '''
    Given a dataframe, explores the basic shape, columns, and information about
    the dataframe.

    Inputs: df (DataFrame) - the dataframe to explore
    '''
    print("Dataframe shape:", df.shape)
    print("Dataframe columns:", df.columns)
    print(df.info())
    print(df.describe())


def explore_data_stats(df, num_cols, time_cols):
    '''
    Given a dataframe, generates histograms for all int / float type columns.
    Also explores the min and max of any datetime columns.

    Inputs:
        df (DataFrame) - the dataframe to explore
        num_cols (lst) - list of column names with int or float type
        time_cols (lst) - list of column names with datetime type
    '''

    for num_col in num_cols:
        df[num_col].hist()

    for dt_col in time_cols:
        print("Min datetime:", df[dt_col].min())
        print("Max datetime:", df[dt_col].max())


def split_data(df, feature_cols, target_cols, test_size=0.2):
    '''
    Splits the data into training and test sets for the feature and target data.

    Inputs:
        df (DataFrame) - the original dataframe
        feature_cols (list) - list of feature columns (columns used to predict)
        target_cols (list) - list of target columns (columns want to predict)
        test_size (float) - the size of the test set (default 0.2)
    Returns:
        (X_train, X_test, y_train, y_test) - tuple which contains the 4 subsets
            of the data.
    '''

    X_train, X_test, y_train, y_test = train_test_split(df[feature_cols],
                                                    df[target_cols],
                                                    test_size=test_size, 
                                                    random_state=5)
    return (X_train, X_test, y_train, y_test)


def impute_missing(df, continuous_col):
    '''
    Impute missing values of continuous variables using the median value.
    This function is used for when missing values are NAN.

    Inputs:
        df (DataFrame) - the original dataframe
        continuous_cols (lst) - list of column names that are continuous 
    Returns:
        df - the dataframe with imputed missing values
    '''

    median_val = df[df[continuous_col].notna()][continuous_col].median()
    df.loc[:, continuous_col] = df[continuous_col].fillna(median_val)
    
    return df


def impute_missing_specialna(df, continuous_col, nan_value):
    '''
    Impute missing values of continuous variables using the median value.
    This function is used for when missing values are a value other than NAN
    (ex: -66666, -99999).

    Inputs:
        df (DataFrame) - the original dataframe
        continuous_cols (lst) - list of column names that are continuous 
    Returns:
        df - the dataframe with imputed missing values
    '''

    median_val = df[df[continuous_col] != nan_value][continuous_col].median()
    df.loc[:, continuous_col] = df[continuous_col].replace(nan_value,
                                                           median_val)
    
    return df


def normalize(df, scaler=None):
    '''
    If scaler is not none, use given scaler's means and sds to normalize 
    (used for test set case).

    Inputs:
        df (DataFrame) - the dataframe to be normalized
        scaler - the scaler to use for normalization
    Returns:
        df (DataFrame) - the normalized dataframe
        scaler - the scaler to be used for next normalization (if needed)
    '''
    # Will not normalize the response (or outcomes),
    # only the predictors (features)

    #Normalizing train set
    if(scaler is None):
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(df) 
    #Normalizing test set
    else:
        normalized_features = scaler.transform(df)

    normalized_df = pd.DataFrame(normalized_features)
    normalized_df.index=df.index
    normalized_df.columns=df.columns

    return normalized_df, scaler


def one_hot(df):
    '''
    Perform one-hot encoding of categorical variables.

    Inputs:
        df (DataFrame) - the initial DataFrame (with only categorical columns)
    Returns:
        df_hot (DataFrame) - the resulting dataframe after 1-hot encoding
    '''

    df_hot = pd.get_dummies(df)
    return df_hot


def dicretize(df, cont_columns):
    '''
    Discretizes continuous variables in a dataframe.

    Inputs:
        df (DataFrame) - the initial DataFrame
        cont_columns (lst) - list of column names that are continuous
    Returns:
        disc_df (DataFrame) - resulting discretized dataframe 
    '''

    disc_df = pd.cut(df[cont_columns])
    return disc_df


def linear_classifier(train_features, train_targets, test_features):
    '''
    Simple function that trains a linear regression model.

    Inputs:
        train_features (DataFrame) - training dataframe with the feature columns
        train_targets (DataFrame) - training dataframe with the target columns
        test_features (DataFrame) - testing dataframe with the feature columns
    Returns:
        target_predict (array) - a numpy array of predicted values based on the
            model and test features 
    '''
    regr = linear_model.LinearRegression()
    start = datetime.datetime.now()
    regr.fit(train_features,train_targets)
    target_predict = regr.predict(np.array(test_features))
    stop = datetime.datetime.now()
    print("Time Elapsed:", stop-start)
    return target_predict


def eval_classifier(test_targets, target_predict, train_features, train_targets):
    '''
    Evaluated the model by computing the MSE, RSS, variance score, and accuracy
    score.

    Inputs:
        test_targets (DataFrame) - testing dataframe with the target columns
        target_predict (array) - a numpy array of predicted values based on the
            model and test features 
        train_features (DataFrame) - training dataframe with the feature columns
        train_targets (DataFrame) - training dataframe with the target columns
    '''
    
    # The mean squared error and RSS (by hand)
    print("Mean squared error:", np.mean((target_predict - test_targets) ** 2))
    print("RSS:", np.sum((target_predict - test_targets) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score:', regr.score(train_features, train_targets))
    print('Accuracy score:', accuracy_score(test_targets, target_predict))


def grid_search(df_train, df_test):
    '''
    Trains and evaluates multiple Machine Learning models.

    Inputs:
        df_train (DataFrame) - the training data (including feature and target
            columns)
        df_test (DataFrame) - the testing data (including feature and target
            columns)
    Returns:
        results (DataFrame) - a dataframe with the models, parameters, accuracy
            scores, mean squared errors, and residual sum of squares
    '''
    # Begin timer 
    start = datetime.datetime.now()
    counter = 0

    # Initialize results df
    results = pd.DataFrame(columns=['model',
                                    'params',
                                    'accuracy',
                                    'mse',
                                    'rss'])

    # Loop over models 
    for model_key in MODELS.keys(): 
    
        # Loop over parameters 
        for params in GRID[model_key]: 
            print("Model Number:", counter)
            print("Training model:", model_key, "|", params)
        
            # Create model 
            model = MODELS[model_key]
            model.set_params(**params)
        
            # Fit model on training set 
            model.fit(np.array(df_train.iloc[:, :-1]),
                      np.array(df_train.iloc[:, -1]))

            # Predict on testing set 
            target_predict = model.predict(np.array(df_test.iloc[:, :-1]))
        
            # Evaluate predictions
            # The mean squared error and RSS (by hand)
            mse = np.mean((target_predict - df_test.iloc[:, -1]) ** 2)
            print("Mean squared error:", mse)
            rss = np.sum((target_predict - df_test.iloc[:, -1]) ** 2)
            print("RSS:", rss)
            accuracy = accuracy_score(df_test.iloc[:, -1], target_predict)
            print("Accuracy score:", accuracy)
            if model_key != "GaussianNB":
                print(model.coef_)
            print()

            # Store results in your results data frame
            results = results.append({'model':model_key,
                                      'params':params,
                                      'accuracy':accuracy,
                                      'mse':mse,
                                      'rss':rss},
                                      ignore_index=True)
            counter +=1
    # End timer
    stop = datetime.datetime.now()
    print("Time Elapsed:", stop - start)
    return results

# Config: Dictionaries of models and hyperparameters
MODELS = {
    'LogisticRegression': LogisticRegression(), 
    'LinearSVC': LinearSVC(), 
    'GaussianNB': GaussianNB()
}

GRID = {
    'LogisticRegression': [{'penalty': x,
                            'C': y,
                            'random_state': 0,
                            'solver':'lbfgs',
                            'max_iter': 1000} 
                           for x in ('l2', 'none') \
                           for y in (0.01, 0.1, 1, 10, 100)],
    'GaussianNB': [{'priors': None}],
    'LinearSVC': [{'C': x, 'random_state': 0} \
                  for x in (0.01, 0.1, 1, 10, 100)]
}
