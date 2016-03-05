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

################################################################################
# Text Utilities ###############################################################
################################################################################

def get_bow():
    """
    Parse keywords into bag of words representation

    Returns
    -------
    Nothing Currently, but can return c for the bag of words with word count and
    return words for the keywords split into tokens

    Examples
    --------

    """
    import os,re
    import pandas as pd
    from collections import Counter
    count = 1
    raw_text = ""
    data = pd.read_json(os.getcwd() + "/data/datafinal_60-16.json")
    numpy_data = np.array(data.iloc[:, 5].values)
    for movie in numpy_data:
        for keyword in movie["keywords"]:
            raw_text += keyword["name"] + " "
            count += 1

    raw_text = str(raw_text).lower()
    reg1 = re.compile('[.,?:;()!\[\]/]')
    raw_text = reg1.sub(' ', raw_text)
    reg3 = re.compile('\'')
    raw_text = reg3.sub('', raw_text)
    reg2 = re.compile('[\s]+')
    raw_text = reg2.sub(' ', raw_text)
    reg3 = re.compile('\'')
    raw_text = raw_text.strip()
    words = raw_text.split()
    words = remove_stopwords(words)
    c = Counter(words)
    return words
    # X = np.array(data.iloc[:,3].values)
    # Y = np.array(data.iloc[:,4].values)
    # for budget in data.iloc[:,3].values:
    #     X.append(budget)
    # print(len(X))
    # for revenue in data.iloc[:,4].values:
    #     Y.append(revenue)
    # print(len(Y))


    # X_tr = X[:2400]
    # X_te = X[2400:]
    # Y_tr = Y[:2400]
    # Y_te = Y[2400:]
    #
    # print(len(X_tr))
    # print(len(X_te))
    # print(len(Y_tr))
    # print(len(Y_te))
    #
    # regr = lm.LinearRegression()
    # regr.fit(X,Y)
    #
    # print("Score: " + str(regr.score(X_te,Y_te)))
    # print(numpy_data)

def remove_stopwords(tokens):
    """
    Removes the a static set of stopwords from the given tokens using remove_stopwords_inner.

    This function should remove the 100 most common English words from the given tokens
    Parameters
    ----------
    tokens : Iterable[str]
        The tokens from which to remove stopwords.

    Returns
    -------
    List[str]
        The input tokens with stopwords removed.

    Example
    -------
    >>> document = load_hamlet()
    >>> tokens = tokenize_simple(document)
    >>> remove_stopwords(tokens)[:10]
    ['tragedie', 'hamlet', 'william', 'shakespeare', '1599', 'actus', 'primus', 'scoena', 'prima', 'enter']
    """
    # the stopwords to remove
    stopwords = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for',
                 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by',
                 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all',
                 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get',
                 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him',
                 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',
                 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
                 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
                 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day',
                 'most', 'us', 'title']
    tokens = [word for word in tokens if word not in stopwords]
    #print(tokens)
    return tokens