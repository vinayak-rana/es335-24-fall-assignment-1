"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    
    pass

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if y.nunique()>=10:
        return 1
    else:
        return 0


def entropy(y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    frequency=y.value_counts()
    p=frequency/len(y)

    return -np.sum(p * np.log2(p + np.finfo(float).eps))   



def gini_index(y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    frequency = y.value_counts()
    p = frequency/len(y)

    return 1-np.sum(p**2)


def information_gain(X:pd.DataFrame,y: pd.Series, feature, value  ,discrete=None) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    info_gain = None
    if discrete:
        y_left = y[X[feature]==value]
        y_right = y[X[feature]!=value]
    
    else:
        y_left = y[X[feature]<=value]   
        y_right = y[X[feature]>value]

    
    info_gain = entropy(y) - ((len(y_left)/len(y)) * entropy(y_left) + (len(y_right)/len(y)) * entropy(y_right))
    

    return info_gain



def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series)->tuple:
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

    pass


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value, discrete = True):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """
    x_left,x_right,y_left,y_right = None, None , None, None

    if discrete:
        left_mask = X[attribute] == value
        right_mask = X[attribute] != value

    else:
        left_mask = X[attribute] <= value
        right_mask = X[attribute] > value

    x_left = X[left_mask]
    x_right = X[right_mask]

    y_left = y[left_mask]
    y_right = y[right_mask]

    return x_left,y_left,x_right,y_right

    



