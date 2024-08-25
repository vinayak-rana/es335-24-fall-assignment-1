from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size,'The predicted labels must be of same size as that of true labels'
    
    assert y_hat.size != 0     ,'The predicted labels cannot be an empty array'

    true_predictions  = (y_hat == y).sum()
    total_predictions = y.size

    return true_predictions/total_predictions


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size,'The predicted labels must be of same size as that of true labels'

    assert y_hat.size != 0     ,'The predicted labels cannot be an empty array'

    true_pos = ((y == cls) and (y_hat == cls)).sum()
    pred_pos = (y_hat == cls).sum()

    if pred_pos == 0:
        return 0
    return true_pos/pred_pos


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size,'The predicted labels must be of same size as that of true labels'

    assert y_hat.size != 0     ,'The predicted labels cannot be an empty array'

    true_pos  = ((y_hat == cls) and (y == cls)).sum()
    total_pos = (y == cls).sum()

    if total_pos == 0:
        return 0
    return true_pos/total_pos


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size,'The predicted labels must be of same size as that of true labels'

    assert y_hat.size != 0     ,'The predicted labels cannot be an empty array'

    return np.sqrt(np.mean((y-y_hat)**2)) 


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size,'The predicted labels must be of same size as that of true labels'

    assert y_hat.size != 0     ,'The predicted labels cannot be an empty array'

    return np.mean(np.abs(y-y_hat))
