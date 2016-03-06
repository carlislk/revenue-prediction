################################################################################
# IMPORTS ######################################################################
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import mltools as ml
import utils
import json
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import os
import pandas as pd
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

def get_feature_lists():
     """
     Parses json file and returns all of the possible features as lists

     :return:
     A list of features with indices:
     0: budget list
     1: inflated budget list
     2: keyword list
     3: overview list
     4: genre list
     5: revenue list
     6: inflated revenue list
     """

     with open(os.getcwd() + "/data/datafinal_60-16.json") as data_file:
        data = json.load(data_file)

     revenue_list = []
     keyword_list = []
     overview_list = []
     genre_list = []
     infrev_list = []
     infbud_list = []
     budget_list = []

     for movie in data:
         if len(movie["keywords"]["keywords"]) > 0 and int(movie["inf_revenue"]) > 1000000 and int(movie["inf_budget"]) > 1000000:
             keywords = ""
             genres = ""

             # builds up this movie's keyword string
             for kw in movie["keywords"]["keywords"]:
                keywords += kw["name"] + " "

             # builds up this movie's genre string
             for genre in movie["genres"]:
                genres += genre["name"] + " "

             # append the various lists with this movie's info
             budget_list.append(str(movie["budget"]))
             infbud_list.append(str(movie["inf_budget"]))
             keyword_list.append(keywords)
             overview_list.append(movie["overview"])
             genre_list.append(genres)
             revenue_list.append(str(movie["revenue"]))
             infrev_list.append(str(movie["inf_revenue"]))


     return [budget_list, infbud_list, keyword_list, overview_list, genre_list, revenue_list, infrev_list]

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

def get_word_set(file_name):
    """
    - Open JSON file and get unique set of words

    Parameters
    ----------
    file_name: String

    Returns
    -------
    X: Features
    Y: Target Values

    Examples
    --------
    >>> split_numpy(data, train_fraction=0)
    tuple(X, Y)

    """

def extract_text_features():
    """
    - Open JSON file and get unique set of words

    Parameters
    ----------
    file_name: String

    Returns
    -------
    X: Features
    Y: Target Values

    Examples
    --------
    >>> split_numpy(data, train_fraction=0)
    tuple(X, Y)

    """

def text_features_grouped(file_name):
    """
    - Open JSON file and get unique set of words

    Parameters
    ----------
    file_name: String

    Returns
    -------
    None

    Examples
    --------
    >>> text_features_grouped(file_name)
    tuple(X, Y)

    """
    print("--------Grouped Features---------\n")

    from sklearn.feature_extraction.text import CountVectorizer



    with open(os.getcwd() + "/data/datafinal_60-16.json") as data_file:
        data = json.load(data_file)


    # KeyWord Features
    keyWords = []
    # Budget Feature
    budgets = []
    # Revenue Target Value
    revenues = []

    for movie in data:
        # Filter Out Data
        if len(movie["keywords"]["keywords"]) > 0 and int(movie["inf_revenue"]) > 1000000 and int(movie["inf_budget"]) > 1000000:

            # Append To KeyWord Features
            stringKeywords = ""
            for kw in movie["keywords"]["keywords"]:
                stringKeywords = stringKeywords + " " +  kw["name"]
            keyWords.append(stringKeywords)
            # Append to Budget
            budgets.append(float(movie["inf_budget"]))
            # Append To Target Value
            revenues.append(float(movie["inf_revenue"]))


    # print(len(keyWords))
    # print(len(budgets))
    # print(len(revenues))


    count_vectorizer = CountVectorizer()
    counts = count_vectorizer.fit_transform(keyWords)

    #print(counts.shape)
    #print(type(counts))


    bd = np.array(budgets)
    rev = np.array(revenues)

    rev = rev[:,np.newaxis]

    #print(kw.shape)
    #print(bd.shape)
    #print(type(rev[0][0]))

    ca = counts.toarray()



    #print(counts[3].shape)

    # # We create the model.
    regr = lm.LinearRegression()

    # We train the model on our training dataset.
    regr.fit(ca, rev)

    print("REV SHAPE: ", rev.shape)
    print("Count Array Shape: ", ca.shape)


    #print("Train MSE   : ", format(np.mean((regr.predict(rev) - rev) ** 2),'f'))

    #yhat = regr.predict(rev)

    #print(type(yhat))

    #print("Test Score  : ", str(regr.score(counts, rev)))


if __name__ == '__main__':
    print()