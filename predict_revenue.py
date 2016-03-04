################################################################################
# IMPORTS #####################################################################
################################################################################

import utils
import json
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import os
import pandas as pd
import mltools as ml


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

def numpy_from_json(file_name, feature_list=[], train_fraction=0):
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
    if len(feature_list) == 0:
        numpy_data = np.array(data.values)
    else:
        numpy_data = np.array(data.iloc[:, feature_list].values)


    # Split & Return Numpy Array -> X(Features), Y(Target Values)
    # Specify fraction parameter to split into train/test
    return utils.split_numpy(numpy_data, train_fraction)

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

def linear_regress_Split_with_Plot(X_tr, Y_tr, X_te, Y_te, title):
    """
    Simple Linear Regression Model Using SciKit Learn Module

    Parameters
    ----------
    X_tr: Features Train
    Y_tr: Train Target Values

    X_te: Features Test
    Y_te: Test Target Values

    title: Description Of Data

    Returns
    -------
    Nothing Currently

    Examples
    --------
    >>> linear_regress_Split_with_Plot(X_tr, Y_tr, X_te, Y_te, title)
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


    ## Plot Train Data
    # Visualises dots, where each dot represent a data exaple and corresponding teacher
    plt.scatter(X_tr, Y_tr,  color='black', marker='+')

    # Plots the linear model
    plt.plot(X_tr, regr.predict(X_tr), color='blue', linewidth=3);
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

    #######################################################################################
    # Print Error Rates ###################################################################
    #######################################################################################

    # The mean square error
    print(title)
    print("Training error: ", format(np.mean((regr.predict(X_tr) - Y_tr) ** 2),'f'))
    print("Test     error: ", format(np.mean((regr.predict(X_te) - Y_te) ** 2), 'f'))

    print("Model Score: " + str(regr.score(X_tr, Y_tr)))


################################################################################
# Experiments ##################################################################
################################################################################

def experiment_1():
    """
    Simple Experiment 1

    Scatter X, Y
        Where X is ONLY one feature.

    """

    # Message
    print("\n----------- Experiment 1 --------------\n")

    # Create Features -> X & Target Values -> Y
    X, Y = numpy_from_json("/data/datafinal_60-16.json", [3,4])

    # Scatter Initial Data
    #scatter_data(X, Y)

    # Scatter Log Base 10 of Data
    #scatter_data(np.log10(X), np.log10(Y))

    # Scatter Rounded To Nearest Int Log Base 10 of Data
    #scatter_data(np.rint(np.log10(X)), np.rint(np.log10(Y)))

    # Scatter Rounded To 1 Decimal Place Log Base 10 of Data
    #scatter_data(np.round(np.log10(X), 1), np.round(np.log10(Y), 1))


    print(X.shape)
    print(Y.shape)

    # Message
    print("\n----------- End 1 ----------------------\n")

def experiment_2():
    """
    Simple Experiment 2

    Test Linear Regression with No Train/Test Split

    Single Feature X

    """

    # Message
    print("\n----------- Experiment 2 ---------------\n")

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

def experiment_3():
    """
    Simple Experiment 3

    Test Linear Regression With Train/Test Split

    Single Feature X

    """

    # Message
    print("\n----------- Experiment 3 ---------------\n")

    # Create Features -> X & Target Values -> Y
    X_tr, X_te, Y_tr, Y_te = numpy_from_json("/data/datafinal_60-16.json", [3,4], 0.75)

    # Shape
    #print(X.shape, Y.shape)

    # Call Linear Regress With Train/Split
    # Log Base 10 Data
    linear_regress_Split_with_Plot(np.log10(X_tr), np.log10(Y_tr), np.log10(X_te), np.log10(Y_te)
                                   ,"Error Rates On Log Base 10 Data")


    # Message
    print("\n----------- End 3 ----------------------\n")


if __name__ == '__main__':

    # In Main Code
    # Call Experiments From Here

    # Plot Examples
    #experiment_1()

    # Linear Regression No Split
    #experiment_2()

    # Linear Regression With Split
    #experiment_3()

    print()
