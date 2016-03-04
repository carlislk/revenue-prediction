################################################################################
# IMPORTS ######################################################################
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import mltools as ml

################################################################################
# General Utilities ############################################################
################################################################################

def split_numpy(data, train_fraction=0):
    """
    - Split Numpy Array Into Features & Target Values
    - Verify Shape Of Features & Target Values

    Parameters
    ----------
    data: Numpy Array
    train_fraction : Fraction that specifies train/test split
        If 0 no split Occurs and function returns only X,Y

    Returns
    -------
    X: Features
    Y: Target Values

    Examples
    --------
    >>> split_numpy(data, train_fraction=0)
    tuple(X, Y)

    """

    # Split Into Features & Target Values
    X = data[:,0:-1]                     # N-1 Columns Of Data ( Scalar Features -> X Values)
    Y = data[:,-1]                       # Last Column of Data ( Target Values -> Y values)

    # Assert Shape
    if len(X.shape) != 2:
        X = X[:,np.newaxis]             # Code expects shape (M,N) so make sure it's 2-dimensional
    if len(Y.shape) != 2:
        Y = Y[:,np.newaxis]             # Code expects shape (M,N) so make sure it's 2-dimensional

    assert(len(X.shape) == 2 )
    assert(len(Y.shape) == 2 )

    # Split Data into Test & Train
    if train_fraction != 0:

        # Split data into 75/25 train/test
        X_train,X_test,Y_train,Y_test = ml.splitData(X,Y, train_fraction)

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
        #print("Train Shape (x, y): (",X_train.shape,",",Y_train.shape,")")
        #print("Test  Shape (x, y): (",X_test.shape,",",Y_test.shape,")")

        return X_train,X_test,Y_train,Y_test

    # If no test/train split return data
    return X,Y

