################################################################################
# IMPORTS ######################################################################
################################################################################

import utils
import json
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import os
import pandas as pd
import mltools as ml
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split

################################################################################
# Data Setup ###################################################################
################################################################################

def pandas_sample(file_name):
    """
    Test Out Pandas Features

    Parameters
    ----------
    file_name: string
        Name of JSON file containing data.

    Returns
    -------
    None

    Examples
    --------
    >>> pandas_sample("/data/datafinal_60-16.json")
    Numpy Array

    Data From datafinal_60-16.json is formatted as follows:
    Index(['budget', 'genres', 'imdb_id', 'inf_budget', 'inf_revenue', 'keywords',
       'overview', 'revenue', 'title'],
      dtype='object')

    """

    # Create Panda Object From JSON
    data = pd.read_json(os.getcwd() + file_name)

    #print(data.dtypes)                         # All Data with Column Headers
    #print(data.head())                         # Overview of Data
    #print(data.index)                          # Length of Data
    #print(data.columns)                        # List of Data Column Names
    #print(data.describe)                       # Overview of Data
    #print(data['budget']                       # Index by Column Name
    #print(data[0:])                            # Print First Column
    #print(data.iloc[3:7,1:2])                  # Print Sliced Data Rows 3-6 and Column 1
    #print(data.iloc[[1,2,4],[0,2]])            # Print Data Rows 1, 2, 4 Column 0 & 2
    #print(data.iloc[1:20, [0, 3, -2, 4]])      # Print Data Sliced Rows, Specified Columns

    print(data[(data["inf_budget"] > 1000000) & (data["inf_revenue"] > 1000000)].iloc[:, [3,4]])
    #print(data["keywords"])

def numpy_from_json(file_name, feature_list=[], train_fraction=0.0):
    """
    Create Numpy Array From JSON File

    Parameters
    ----------
    file_name: string
        Name of JSON file containing data.
    feature_list: list of index's of fetures to select
        If empty all data is selected
    train_fraction: fraction to split data train/test
        If empty 0, no split

    Returns
    -------
    Tuple of Numpy Arrays
        If train_fraction not specified X, Y is returned
        Else X_tr, X_te, Y_tr, Y_te is returned

    Examples
    --------
    All Features Are Used
    >>> X, Y = numpy_from_json("/data/datafinal_60-16.json")
    X, Y

    Specified Features Are Used
     >>> X, Y = numpy_from_json("/data/datafinal_60-16.json", [3,4])
    X, Y

    Specified Features Are Used & Data is Split into Train/Test based off fraction
     >>> X, Y = numpy_from_json("/data/datafinal_60-16.json", [3,4], 0.75)
    X_tr, X_te, Y_tr, Y_te

    Data From datafinal_60-16.json is formatted as follows:
    Index(['budget', 'genres', 'imdb_id', 'inf_budget', 'inf_revenue', 'keywords',
       'overview', 'revenue', 'title'],
      dtype='object')

    """

    # Create Panda Frame From JSON
    data = pd.read_json(os.getcwd() + file_name)

    # Create Numpy Array From Panda Frame
    # Select features based on parameter feature_list
    # If empty select all features
    if len(feature_list) == 0.0:
        numpy_data = np.array(data.values)
    else:
        numpy_data = np.array(data[(data["inf_budget"] > 1000000) & (data["inf_revenue"] > 1000000)].iloc[:, feature_list].values)


    # Split & Return Numpy Array -> X(Features), Y(Target Values)
    # Specify fraction parameter to split into train/test
    return utils.split_numpy(numpy_data, train_fraction)

def data_config():
    """
    Setup Data From Pandas Object

    """
    print("Data Config\n\n")
    #pandas_sample("/data/datafinal_60-16.json")
    utils.text_features_grouped("/data/datafinal_60-16.json")

################################################################################
# Plots ########################################################################
################################################################################

def scatter_data(X, Y):
    """
    Scatter Data

    Parameters
    ----------
    X: Features
    Y: Target Values

    Returns
    -------
    Nothing Currently

    Examples
    --------
    >>> scatter_data(X, Y)
    Nothing Currently

    """

    plt.title("Scatter Initial Data")
    plt.scatter(X, Y)
    plt.xlabel('Budget')
    plt.ylabel('Revenue')
    plt.show()

################################################################################
# Models #######################################################################
################################################################################

def linear_regress_No_Split_with_Plot(X, Y):
    """
    Simple Linear Regression Model Using SciKit Learn Module

    Parameters
    ----------
    X: Features
    Y: Target Values

    Returns
    -------
    Nothing Currently

    Examples
    --------
    >>> linear_regress_No_Split_with_Plot(X, Y)
    Nothing Currently

    Data From datafinal_60-16.json is formatted as follows:
    Index(['budget', 'genres', 'imdb_id', 'inf_budget', 'inf_revenue', 'keywords',
       'overview', 'revenue', 'title'],
      dtype='object')

    """

    # We create the model.
    regr = lm.LinearRegression()

    # We train the model on our training dataset.
    regr.fit(X, Y)

    # Print Coefficients & Intercept (Bias)
    # print()
    # print("Coefficients: ", regr.coef_)
    # print("Intercept (Bias): ",regr.intercept_)

    # Visualises dots, where each dot represent a data exmaple and corresponding teacher
    plt.scatter(X, Y,  color='black', marker='+')

    # Plots the linear model
    plt.plot(X, regr.predict(X), color='blue', linewidth=3)
    plt.title('Linear Regression: Data B - Log Base 10 Raw Budget')
    plt.xlabel('Budget')
    plt.ylabel('Target - Revenue')
    plt.show()

def linear_regress_Split_with_Plot(X_tr, Y_tr, X_te, Y_te, title, plot=False):
    """
    Simple Linear Regression Model Using SciKit Learn Module

    Parameters
    ----------
    X_tr: Features Train
    Y_tr: Train Target Values

    X_te: Features Test
    Y_te: Test Target Values

    title: Description Of Data

    plot: Boolean - Show Plots

    Returns
    -------
    Nothing Currently

    Examples
    --------
    >>> linear_regress_Split_with_Plot(X_tr, Y_tr, X_te, Y_te, title, plot=False)
    Nothing Currently

    Data From datafinal_60-16.json is formatted as follows:
    Index(['budget', 'genres', 'imdb_id', 'inf_budget', 'inf_revenue', 'keywords',
       'overview', 'revenue', 'title'],
      dtype='object')

    """

    # We create the model.
    regr = lm.LinearRegression()

    # We train the model on our training dataset.
    regr.fit(X_tr, Y_tr)

    # Print Coefficients & Intercept (Bias)
    # print()
    # print("Coefficients: ", regr.coef_)
    # print("Intercept (Bias): ",regr.intercept_)


    if plot:
        ## Plot Train Data
        # Visualises dots, where each dot represent a data example and corresponding teacher
        plt.scatter(X_tr, Y_tr,  color='black', marker='+')

        # Plots the linear model
        plt.plot(X_tr, regr.predict(X_tr), color='blue', linewidth=3)
        plt.title('Linear Regression: Train Data - Log Base 10 Raw Budget')
        plt.xlabel('Budget')
        plt.ylabel('Target - Revenue')
        plt.show()

        ## Plot Test Data
        # Visualises dots, where each dot represent a data exaple and corresponding teacher
        plt.scatter(X_te, Y_te,  color='black', marker='+')

        # Plots the linear model
        plt.plot(X_te, regr.predict(X_te), color='blue', linewidth=3)
        plt.title('Linear Regression: Test Data - Log Base 10 Raw Budget')
        plt.xlabel('Budget')
        plt.ylabel('Target - Revenue')
        plt.show()

    ## Print Error Rates
    # The mean square error
    print(title)
    print("Train MSE   : ", format(np.mean((regr.predict(X_tr) - Y_tr) ** 2),'f'))
    print("Test  MSE   : ", format(np.mean((regr.predict(X_te) - Y_te) ** 2), 'f'))

    print("Train Score : ", str(regr.score(X_tr, Y_tr)))
    print("Test Score  : ", str(regr.score(X_te, Y_te)))

################################################################################
# Regression Experiments #######################################################
################################################################################

def single_class_1():
    """
    Simple Experiment 1

    Scatter X, Y
        Where X is ONLY one feature.

    """

    # Message
    print("\n----------- Single Class 1 --------------\n")

    # Create Features -> X & Target Values -> Y
    X, Y = numpy_from_json("/data/datafinal_60-16.json", [3,4])

    # Scatter Initial Data
    #scatter_data(X, Y)

    # Scatter Log Base 10 of Data
    scatter_data(np.log10(X), np.log10(Y))

    # Scatter Rounded To Nearest Int Log Base 10 of Data
    #scatter_data(np.rint(np.log10(X)), np.rint(np.log10(Y)))

    # Scatter Rounded To 1 Decimal Place Log Base 10 of Data
    #scatter_data(np.round(np.log10(X), 1), np.round(np.log10(Y), 1))


    print(X.shape)
    print(Y.shape)

    # Message
    print("\n----------- End 1 ----------------------\n")

def single_class_2():
    """
    Simple Experiment 2

    Test Linear Regression with No Train/Test Split

    Single Feature X

    """

    # Message
    print("\n----------- Single Class 2 ---------------\n")

    # Create Features -> X & Target Values -> Y
    X, Y = numpy_from_json("/data/datafinal_60-16.json", [3,4])

    # Shape
    #print(X.shape, Y.shape)

    # Call Linear Regress With no Train/Split
    # Raw Data
    #linear_regress_No_Split_with_Plot(X, Y)

    # Call Linear Regress With no Train/Split
    # Log Base 10 Data
    linear_regress_No_Split_with_Plot(np.log10(X), np.log10(Y))

    # Message
    print("\n----------- End 2 ----------------------\n")

def single_class_3():
    """
    Simple Experiment 3

    Test Linear Regression With Train/Test Split

    Single Feature X

    Data Log Base 10 Rounded to Nearest Int

    """

    # Message
    print("\n----------- Single Class 3 ---------------\n")

    # Create Features -> X & Target Values -> Y
    X_tr, X_te, Y_tr, Y_te = numpy_from_json("/data/datafinal_60-16.json", [3,4], 0.75)

    # Shape
    #print(X.shape, Y.shape)

    # Call Linear Regress With Train/Split
    # Log Base 10 Data
    linear_regress_Split_with_Plot(np.round(np.log10(X_tr), 0), np.round(np.log10(Y_tr),0)
                                   , np.round(np.log10(X_te), 0), np.round(np.log10(Y_te),0)
                                   ,"Error Rates On Log Base 10 Data")


    # Message
    print("\n----------- End 3 ----------------------\n")

def single_class_4(plot):
    """
    Simple Experiment 4

    Test Linear Regression With Train/Test Split

    Single Feature X

    Data: Raw Budget & Revenue: Not adjusted for Inflation
        1. Raw Data
        2. Log 10 Data

    """

    # Message
    print("\n----------- Single Class 4 ---------------\n")
    print(" - Budget & Revenue Not adjusted For Inflation\n")

    # Create Features -> X & Target Values -> Y
    X_tr, X_te, Y_tr, Y_te = numpy_from_json("/data/datafinal_60-16.json", [0, -2], 0.75)

    # Shape
    #print(X.shape, Y.shape)

    # Call Linear Regress With Train/Split
    # Raw Data
    linear_regress_Split_with_Plot(X_tr, Y_tr, X_te, Y_te
                                   ,"Error Rates On Raw Data", plot)

    print()
    print()

    # Call Linear Regress With Train/Split
    # Log Base 10 Data
    linear_regress_Split_with_Plot(np.log10(X_tr), np.log10(Y_tr), np.log10(X_te), np.log10(Y_te)
                                   ,"Error Rates On Log Base 10 Data", plot)

    # Message
    print("\n----------- End 4 ----------------------\n")

def single_class_5(plot):
    """
    Simple Experiment 5

    Test Linear Regression With Train/Test Split

    Single Feature X

    Data: Budget & Revenue Adjusted For Inflation
        1. Raw Data
        2. Log 10 Data

    """

    # Message
    print("\n----------- Single Class 5 ---------------\n")
    print(" - Budget & Revenue Adjusted For Inflation\n")

    # Create Features -> X & Target Values -> Y
    X_tr, X_te, Y_tr, Y_te = numpy_from_json("/data/datafinal_60-16.json", [3,4], 0.75)

    # Shape
    #print(X.shape, Y.shape)

    # Call Linear Regress With Train/Split
    # Raw Data
    linear_regress_Split_with_Plot(X_tr, Y_tr, X_te, Y_te
                                   ,"Error Rates On Raw Data", plot)

    print()
    print()

    # Call Linear Regress With Train/Split
    # Log Base 10 Data
    linear_regress_Split_with_Plot(np.log10(X_tr), np.log10(Y_tr), np.log10(X_te), np.log10(Y_te)
                                   ,"Error Rates On Log Base 10 Data", plot)

    # Message
    print("\n----------- End 5 ----------------------\n")

def regression_single_numeric(title, feature_index, train_split, take_log=False):
    """
    Simple Regression 1

    Using Numeric Features Budget or Budget INF

    A list of features with indices:
        0: budget list
        1: inflated budget list
        2: keyword list
        3: overview list
        4: genre list
        5: revenue list
        6: inflated revenue list

    """
    print("\n---------- Single Feature Numeric Regression --------")
    print("--------", title, "---------\n")

    features_list = utils.get_feature_lists()
    feature_list = features_list[feature_index]
    target_list = features_list[5]

    X_train, X_test, Y_train, Y_test = train_test_split(feature_list,target_list, train_size=train_split, random_state=19)

    # Convert To Numpy Array
    X_train = np.array(X_train).astype(np.float)
    Y_train = np.array(Y_train).astype(np.float)
    X_test = np.array(X_test).astype(np.float)
    Y_test = np.array(Y_test).astype(np.float)

    # Take Log Of Values
    if take_log:
        X_train = np.log10(X_train)
        Y_train = np.log10(Y_train)
        X_test = np.log10(X_test)
        Y_test = np.log10(Y_test)


    # Assert Shape
    if len(X_train.shape) != 2:
        X_train = np.array(X_train[:,np.newaxis])
    if len(X_test.shape) != 2:
        X_test = np.array(X_test[:,np.newaxis])
    if len(Y_train.shape) != 2:
        Y_train = np.array(Y_train[:,np.newaxis])
    if len(Y_test.shape) != 2:
        Y_test = np.array(Y_test[:,np.newaxis])

    assert (len(X_train.shape) == 2)
    assert (len(X_test.shape) == 2)
    assert (len(Y_train.shape) == 2)
    assert (len(Y_test.shape) == 2)


    regr = lm.LinearRegression().fit(X_train,Y_train)
    print(regr.score(X_test,Y_test))

def regression_single_text(title, feature_index, train_split=.75):
    """
    Simple Regression 2

    Text Features

    A list of features with indices:
        0: budget list
        1: inflated budget list
        2: keyword list
        3: overview list
        4: genre list
        5: revenue list
        6: inflated revenue list

    """
    print("\n---------- Single Feature Text Regression --------")
    print("--------", title, "---------\n")

    features_list = utils.get_feature_lists()
    feature_list = features_list[feature_index]
    target_list = features_list[5]

    X_train, X_test, y_train, y_test = train_test_split(feature_list,target_list, train_size=train_split, random_state=19)

    cv = CountVectorizer()
    X_train_counts = cv.fit_transform(X_train)
    X_test_counts = cv.transform(X_test)

    print('Dimensions of X_train_counts are',X_train_counts.shape)
    print()
    num_non_zero = X_train_counts.nnz
    av_num_word_tokens_per_doc = X_train_counts.sum(axis=1).mean()
    av_num_docs_per_word_token = X_train_counts.sum(axis=0).mean()

    print()
    print('Number of non-zero elements in X_train_counts:', num_non_zero) # 377580
    print('Average number of word tokens per document:', "%.4f" % av_num_word_tokens_per_doc) # 183.4145
    print('Average number of documents per word token:', "%.4f" % av_num_docs_per_word_token) # 13.7569
    print()

    regr = lm.LinearRegression().fit(X_train_counts,np.array(y_train).astype(np.float))
    print(regr.score(X_test_counts,np.array(y_test).astype(np.float)))

def lasso_single_numeric(title, feature_index, train_split, take_log=False):
    """
    Lasso With CV on single numeric Feature


    """
    print("\n---------- Single Feature Numeric Regression --------")
    print("--------", title, "---------\n")

    features_list = utils.get_feature_lists()
    feature_list = features_list[feature_index]
    target_list = features_list[5]

    X_train, X_test, Y_train, Y_test = train_test_split(feature_list,target_list, train_size=train_split, random_state=19)

    # Convert To Numpy Array
    X_train = np.array(X_train).astype(np.float)
    Y_train = np.array(Y_train).astype(np.float)
    X_test = np.array(X_test).astype(np.float)
    Y_test = np.array(Y_test).astype(np.float)

    # Take Log Of Values
    if take_log:
        X_train = np.log10(X_train)
        Y_train = np.log10(Y_train)
        X_test = np.log10(X_test)
        Y_test = np.log10(Y_test)


    # Assert Shape
    if len(X_train.shape) != 2:
        X_train = np.array(X_train[:,np.newaxis])
    if len(X_test.shape) != 2:
        X_test = np.array(X_test[:,np.newaxis])
    if len(Y_train.shape) != 2:
        Y_train = np.array(Y_train[:,np.newaxis])
    if len(Y_test.shape) != 2:
        Y_test = np.array(Y_test[:,np.newaxis])

    assert (len(X_train.shape) == 2)
    assert (len(X_test.shape) == 2)
    assert (len(Y_train.shape) == 2)
    assert (len(Y_test.shape) == 2)


    lasso_model = lm.LassoCV(cv = 15, copy_X = True, normalize=True)
    lasso_fit = lasso_model.fit(X_train,Y_train)
    lasso_path = lasso_model.score(X_test,Y_test)

    print()
    print ('Deg. Coefficient')
    print (pd.Series(np.r_[lasso_fit.intercept_, lasso_fit.coef_]))
    print("Lasso Path: ", lasso_path)

    # print()
    # # Plot the average MSE across folds
    # plt.plot(-np.log(lasso_fit.alphas_),
    # np.sqrt(lasso_fit.mse_path_).mean(axis = 1))
    # plt.ylabel('RMSE (avg. across folds)')
    # plt.xlabel("-log(lambda)")
    # # Indicate the lasso parameter that minimizes the average MSE across
    # plt.axvline(-np.log(lasso_fit.alpha_), color = 'red')
    # plt.show()

def lasso_single_text(title, feature_index, train_split):
    """
    Lasso With CV on single text Feature

    """
    print("\n---------- Single Feature Numeric Regression --------")
    print("--------", title, "---------\n")

    features_list = utils.get_feature_lists()
    feature_list = features_list[feature_index]
    target_list = features_list[5]

    X_train, X_test, Y_train, Y_test = train_test_split(feature_list,target_list, train_size=train_split, random_state=19)

    # Convert To Numpy Array
    Y_train = np.array(Y_train).astype(np.float)
    Y_test = np.array(Y_test).astype(np.float)

    cv = CountVectorizer()
    X_train_counts = cv.fit_transform(X_train)
    X_test_counts = cv.transform(X_test)

    print('Dimensions of X_train_counts are',X_train_counts.shape)
    print()
    num_non_zero = X_train_counts.nnz
    av_num_word_tokens_per_doc = X_train_counts.sum(axis=1).mean()
    av_num_docs_per_word_token = X_train_counts.sum(axis=0).mean()

    print()
    print('Number of non-zero elements in X_train_counts:', num_non_zero) # 377580
    print('Average number of word tokens per document:', "%.4f" % av_num_word_tokens_per_doc) # 183.4145
    print('Average number of documents per word token:', "%.4f" % av_num_docs_per_word_token) # 13.7569
    print()


    lasso_model = lm.LassoCV(cv = 2, copy_X = True, normalize=True)
    lasso_fit = lasso_model.fit(X_train_counts,Y_train)
    lasso_path = lasso_model.score(X_test_counts,Y_test)

    print()
    print ('Deg. Coefficient')
    print (pd.Series(np.r_[lasso_fit.intercept_, lasso_fit.coef_]))
    print("Lasso Path: ", lasso_path)

    print()
    # Plot the average MSE across folds
    plt.plot(-np.log(lasso_fit.alphas_),
    np.sqrt(lasso_fit.mse_path_).mean(axis = 1))
    plt.ylabel('RMSE (avg. across folds)')
    plt.xlabel("-log(lambda)")
    # Indicate the lasso parameter that minimizes the average MSE across
    plt.axvline(-np.log(lasso_fit.alpha_), color = 'red')
    plt.show()

def ridge_single_numeric(title, feature_index, train_split, take_log=False):
    """
    Ridge With CV on single numeric Feature


    """
    print("\n---------- Single Feature Numeric Regression --------")
    print("--------", title, "---------\n")

    features_list = utils.get_feature_lists()
    feature_list = features_list[feature_index]
    target_list = features_list[5]

    X_train, X_test, Y_train, Y_test = train_test_split(feature_list,target_list, train_size=train_split, random_state=19)

    # Convert To Numpy Array
    X_train = np.array(X_train).astype(np.float)
    Y_train = np.array(Y_train).astype(np.float)
    X_test = np.array(X_test).astype(np.float)
    Y_test = np.array(Y_test).astype(np.float)

    # Take Log Of Values
    if take_log:
        X_train = np.log10(X_train)
        Y_train = np.log10(Y_train)
        X_test = np.log10(X_test)
        Y_test = np.log10(Y_test)


    # Assert Shape
    if len(X_train.shape) != 2:
        X_train = np.array(X_train[:,np.newaxis])
    if len(X_test.shape) != 2:
        X_test = np.array(X_test[:,np.newaxis])
    if len(Y_train.shape) != 2:
        Y_train = np.array(Y_train[:,np.newaxis])
    if len(Y_test.shape) != 2:
        Y_test = np.array(Y_test[:,np.newaxis])

    assert (len(X_train.shape) == 2)
    assert (len(X_test.shape) == 2)
    assert (len(Y_train.shape) == 2)
    assert (len(Y_test.shape) == 2)


    ridge_model = lm.RidgeCV(cv = 15, normalize=True)
    ridge_fit = ridge_model.fit(X_train,Y_train)
    ridge_path = ridge_model.score(X_test,Y_test)

    print()
    #print ('Deg. Coefficient')
    #print (pd.Series(np.r_[ridge_fit.intercept_, ridge_fit.coef_]))
    print("Lasso Path: ", ridge_path)

    # print()
    # # Plot the average MSE across folds
    # plt.plot(-np.log(lasso_fit.alphas_),
    # np.sqrt(lasso_fit.mse_path_).mean(axis = 1))
    # plt.ylabel('RMSE (avg. across folds)')
    # plt.xlabel("-log(lambda)")
    # # Indicate the lasso parameter that minimizes the average MSE across
    # plt.axvline(-np.log(lasso_fit.alpha_), color = 'red')
    # plt.show()

def ridge_single_text(title, feature_index, train_split):
    """
    Lasso With CV on single text Feature

    """
    print("\n---------- Single Feature Numeric Regression --------")
    print("--------", title, "---------\n")

    features_list = utils.get_feature_lists()
    feature_list = features_list[feature_index]
    target_list = features_list[5]

    X_train, X_test, Y_train, Y_test = train_test_split(feature_list,target_list, train_size=train_split, random_state=19)

    # Convert To Numpy Array
    Y_train = np.array(Y_train).astype(np.float)
    Y_test = np.array(Y_test).astype(np.float)

    cv = CountVectorizer()
    X_train_counts = cv.fit_transform(X_train)
    X_test_counts = cv.transform(X_test)

    print('Dimensions of X_train_counts are',X_train_counts.shape)
    print()
    num_non_zero = X_train_counts.nnz
    av_num_word_tokens_per_doc = X_train_counts.sum(axis=1).mean()
    av_num_docs_per_word_token = X_train_counts.sum(axis=0).mean()

    print()
    print('Number of non-zero elements in X_train_counts:', num_non_zero) # 377580
    print('Average number of word tokens per document:', "%.4f" % av_num_word_tokens_per_doc) # 183.4145
    print('Average number of documents per word token:', "%.4f" % av_num_docs_per_word_token) # 13.7569
    print()


    lr = lm.RidgeCV(cv = 2, normalize=True)
    lr_fit = lr.fit(X_train_counts,Y_train)
    lr_path = lr.score(X_test_counts,Y_test)

    print()
    print ('Deg. Coefficient')
    print (pd.Series(np.r_[lr_fit.intercept_, lr_fit.coef_]))
    print("Lasso Path: ", lr_path)

    # print()
    # # Plot the average MSE across folds
    # plt.plot(-np.log(lr_fit.alphas_),
    # np.sqrt(lr_fit.mse_path_).mean(axis = 1))
    # plt.ylabel('RMSE (avg. across folds)')
    # plt.xlabel("-log(lambda)")
    # # Indicate the lasso parameter that minimizes the average MSE across
    # plt.axvline(-np.log(lr_fit.alpha_), color = 'red')
    # plt.show()

def elastic_single_numeric(title, feature_index, train_split, take_log=False):
    """
    Elastic With CV on single numeric Feature


    """
    print("\n---------- Single Feature Numeric Regression --------")
    print("--------", title, "---------\n")

    features_list = utils.get_feature_lists()
    feature_list = features_list[feature_index]
    target_list = features_list[5]

    X_train, X_test, Y_train, Y_test = train_test_split(feature_list,target_list, train_size=train_split, random_state=19)

    # Convert To Numpy Array
    X_train = np.array(X_train).astype(np.float)
    Y_train = np.array(Y_train).astype(np.float)
    X_test = np.array(X_test).astype(np.float)
    Y_test = np.array(Y_test).astype(np.float)

    # Take Log Of Values
    if take_log:
        X_train = np.log10(X_train)
        Y_train = np.log10(Y_train)
        X_test = np.log10(X_test)
        Y_test = np.log10(Y_test)


    # Assert Shape
    if len(X_train.shape) != 2:
        X_train = np.array(X_train[:,np.newaxis])
    if len(X_test.shape) != 2:
        X_test = np.array(X_test[:,np.newaxis])
    if len(Y_train.shape) != 2:
        Y_train = np.array(Y_train[:,np.newaxis])
    if len(Y_test.shape) != 2:
        Y_test = np.array(Y_test[:,np.newaxis])

    assert (len(X_train.shape) == 2)
    assert (len(X_test.shape) == 2)
    assert (len(Y_train.shape) == 2)
    assert (len(Y_test.shape) == 2)


    lr = lm.ElasticNetCV(cv = 15, normalize=True)
    lr_fit = lr.fit(X_train,Y_train)
    lr_path = lr.score(X_test,Y_test)

    print()
    #print ('Deg. Coefficient')
    #print (pd.Series(np.r_[ridge_fit.intercept_, ridge_fit.coef_]))
    print("Elastic Path: ", lr_path)

    # print()
    # # Plot the average MSE across folds
    # plt.plot(-np.log(lr_fit.alphas_),
    # np.sqrt(lr_fit.mse_path_).mean(axis = 1))
    # plt.ylabel('RMSE (avg. across folds)')
    # plt.xlabel("-log(lambda)")
    # # Indicate the lasso parameter that minimizes the average MSE across
    # plt.axvline(-np.log(lr_fit.alpha_), color = 'red')
    # plt.show()

def elastic_single_text(title, feature_index, train_split):
    """
    Elastic With CV on single text Feature

    """
    print("\n---------- Single Feature Numeric Regression --------")
    print("--------", title, "---------\n")

    features_list = utils.get_feature_lists()
    feature_list = features_list[feature_index]
    target_list = features_list[5]

    X_train, X_test, Y_train, Y_test = train_test_split(feature_list,target_list, train_size=train_split, random_state=19)

    # Convert To Numpy Array
    Y_train = np.array(Y_train).astype(np.float)
    Y_test = np.array(Y_test).astype(np.float)

    cv = CountVectorizer()
    X_train_counts = cv.fit_transform(X_train)
    X_test_counts = cv.transform(X_test)

    print('Dimensions of X_train_counts are',X_train_counts.shape)
    print()
    num_non_zero = X_train_counts.nnz
    av_num_word_tokens_per_doc = X_train_counts.sum(axis=1).mean()
    av_num_docs_per_word_token = X_train_counts.sum(axis=0).mean()

    print()
    print('Number of non-zero elements in X_train_counts:', num_non_zero) # 377580
    print('Average number of word tokens per document:', "%.4f" % av_num_word_tokens_per_doc) # 183.4145
    print('Average number of documents per word token:', "%.4f" % av_num_docs_per_word_token) # 13.7569
    print()


    lr = lm.ElasticNetCV(cv = 2, normalize=True)
    lr_fit = lr.fit(X_train_counts,Y_train)
    lr_path = lr.score(X_test_counts,Y_test)

    print()
    print ('Deg. Coefficient')
    print (pd.Series(np.r_[lr_fit.intercept_, lr_fit.coef_]))
    print("Elastic Path: ", lr_path)

    # print()
    # # Plot the average MSE across folds
    # plt.plot(-np.log(lr_fit.alphas_),
    # np.sqrt(lr_fit.mse_path_).mean(axis = 1))
    # plt.ylabel('RMSE (avg. across folds)')
    # plt.xlabel("-log(lambda)")
    # # Indicate the lasso parameter that minimizes the average MSE across
    # plt.axvline(-np.log(lr_fit.alpha_), color = 'red')
    # plt.show()

################################################################################
# Classification Experiments ###################################################
################################################################################

# None Currently

################################################################################
# Controllers ##################################################################
################################################################################

def single_class_regression():
    """
    Single Class Regression

    Simple Experiments Using A Single Feature for X.
    Want to establish baseline for models and test
    performance on different data transformations
        ex. raw, log10, round(log10, 0), round(log10, 1)

    """

    # Call Experiments From Here

    # Configure Data
    #data_config()

    # Plot Examples
    #single_class_1()

    # Linear Regression No Split
    #single_class_2()

    # Linear Regression With Split - Data Rounded To nearest Int
    #single_class_3()

    # Linear Regression With Split - Raw Budget & Revenue: [ Raw Data, Log Base 10 Data ]
    #single_class_4(False)

    # Linear Regression With Split - Budget & Revenue Adjusted For Inflation: [ Raw Data, Log Base 10 Data ]
    #single_class_5(False)

def simple_regression():
    """
    Simple Regression Models

    Expand Regression Models to include more features.
        A list of features with indices:
        0: budget list
        1: inflated budget list
        2: keyword list
        3: overview list
        4: genre list
        5: revenue list
        6: inflated revenue list



    """
    # Call Experiments From Here

    ## ---------- Linear Regression: Single Feature ------------

    # ## Numeric Features
    # # Budget
    # regression_single_numeric("Budget", 0, .80)
    #
    # # Budget Inflated
    # regression_single_numeric("Budget INF", 1, .80)
    #
    # # Budget Log Base 10
    # regression_single_numeric("Budget Log Base 10", 0, .80, True)
    #
    # # Budget Inflated Log Base 10
    # regression_single_numeric("Budget INF Log Base 10", 1, .80, True)
    #
    #
    # ## Text Features
    # # Keywords
    # regression_single_text("Keywords", 2, .80)
    #
    # # Overview
    # regression_single_text("Overview", 3, .80)
    #
    # # Genre
    # regression_single_text("Genre", 4, .80)


    # ## ---------- CV Lasso Regression: Single Numeric  ---------
    #
    # # Budget
    # lasso_single_numeric("Lasso with CV Budget", 0, .80)
    #
    # # Budget Inflation
    # lasso_single_numeric("Lasso with CV Budget", 1, .80)
    #
    # # Budget Log Base 10
    # lasso_single_numeric("Lasso with CV Budget", 0, .80, True)
    #
    # # Budget Inflation Log Base 10
    # lasso_single_numeric("Lasso with CV Budget", 1, .80, True)
    #
    # ## ---------- CV Lasso Regression: Single Text  ------------
    #
    # # Run But Take Forever CV SET TO 2
    #
    # # Keywords
    # lasso_single_text("Lasso with Keywords", 2, .80)
    #
    # # Overview
    # lasso_single_text("Lasso with Overview", 3, .80)
    #
    # # Genre
    # lasso_single_text("Lasso with Genre", 4, .80)


    # ## ---------- CV Ridge Regression: Single Numeric ----------
    #
    # # Budget
    # ridge_single_numeric("Lasso with CV Budget", 0, .80)
    #
    # # Budget Inflation
    # ridge_single_numeric("Lasso with CV Budget", 1, .80)
    #
    # # Budget Log Base 10
    # ridge_single_numeric("Lasso with CV Budget", 0, .80, True)
    #
    # # Budget Inflation Log Base 10
    # ridge_single_numeric("Lasso with CV Budget", 1, .80, True)

    ## ---------- CV Ridge Regression: Single Text -------------

    # # Run But Take Forever CV SET TO 2
    #
    # # Keywords
    # ridge_single_text("Ridge with Keywords", 2, .80)
    #
    # # Overview
    # ridge_single_text("Ridge with Overview", 3, .80)
    #
    # # Genre
    # ridge_single_text("Ridge with Genre", 4, .80)


    ## ---------- CV Elastic Net Regression: Single Numeric ----------

    # # Budget
    # elastic_single_numeric("Elastic with CV Budget", 0, .80)
    #
    # # Budget Inflation
    # elastic_single_numeric("Elastic with CV Budget", 1, .80)
    #
    # # Budget Log Base 10
    # elastic_single_numeric("Elastic with CV Budget", 0, .80, True)
    #
    # # Budget Inflation Log Base 10
    # elastic_single_numeric("Elastic with CV Budget", 1, .80, True)

    ## ---------- CV Elastic Net Regression: Single Text -------------

    # # Run But Take Forever CV SET TO 2

    # # Keywords
    # elastic_single_text("Elastic with Keywords", 2, .80)
    #
    # # Overview
    # elastic_single_text("Elastic with Overview", 3, .80)
    #
    # # Genre
    # elastic_single_text("Elastic with Genre", 4, .80)


################################################################################
# Main #########################################################################
################################################################################

if __name__ == '__main__':

    # In Main Code
    # Call Controllers From Here

    # Establish Baseline with single feature budget
    #single_class_regression()

    # Using single numeric & text features
    simple_regression()




