"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal
from abc import ABC,abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
@dataclass
class Node:
    """
    Represents a node in the Decision Tree
    """
    def __init__(
            self,
            left_child   = None,
            right_child  = None,
            prediction   = None,
            feature      = None,
            threshhold   = None,
            node_info    =Literal['discrete','real']
    ):
        self.left_child  = left_child
        self.right_child = right_child
        self.prediction  = prediction
        self.feature     = feature
        self.threshhold  = threshhold
        self.node_info   = node_info

class BaseDecisionTree(ABC):
    
    @abstractmethod
    def fit(self,X:pd.DataFrame,y:pd.Series):
        pass

    @abstractmethod
    def predict(self,X:pd.DataFrame):
        pass

    @abstractmethod
    def build_tree(self,X:pd.DataFrame,y:pd.Series,depth):
        pass

    @abstractmethod
    def find_optimal_split(self,X:pd.DataFrame,y:pd.Series)-> tuple:
        pass


class DecisionTreeClassifier(Node,BaseDecisionTree):
    
    def __init__(self,criterion,max_depth=10):
        self.criterion = criterion
        self.root      = None
        self.max_depth = max_depth
        self.is_real   = []

    def build_tree(self,X:pd.DataFrame,y:pd.Series,depth):
        
        if depth>=self.max_depth or len(X)==0 or y.nunique()==1:
            return Node(prediction=y.mode().iloc[0])

        
        split_feature , split_value , info_gain = self.find_optimal_split(X,y)

        if split_feature is None:
            return Node(prediction=y.mode().iloc[0])
        
        if_discrete= not check_ifreal(X[split_feature])
        x_left,y_left,x_right,y_right = split_data(X,y,split_feature,split_value,if_discrete)

        L_child = self.build_tree(x_left,y_left,depth+1)
        R_child = self.build_tree(x_right,y_right,depth+1)

        node_type = 'real' if self.is_real[X.columns.get_loc(split_feature)] else 'discrete'
        return Node(left_child=L_child , right_child=R_child , feature=split_feature , threshhold=split_value , node_info=node_type)


    
    def find_optimal_split(self,X:pd.DataFrame,y:pd.Series)-> tuple:
        best_split_metric = float('-inf') if self.criterion == 'information_gain' else float('inf')
        best_split_feature = None
        best_split_value = None
        for col in X.columns:
            if not check_ifreal(X[col]):
                for split_value in X[col].unique():

                    curr_split_metric = self._calculate_split_metric(X,y,col,split_value,discrete=True)
                    
                    if self._is_best_split(curr_split_metric,best_split_metric):
                        best_split_metric = curr_split_metric
                        best_split_feature = col
                        best_split_value = split_value
            
            else:
                for split_value in self._get_split_values(X[col]):                    
                    curr_split_metric = self._calculate_split_metric(X,y,col,split_value,discrete=False)

                    if self._is_best_split(curr_split_metric,best_split_metric):
                        best_split_metric = curr_split_metric
                        best_split_feature = col
                        best_split_value = split_value
        
        return best_split_feature , best_split_value , best_split_metric

    def fit(self,X,y):
        for col in X.columns:
            self.is_real.append(check_ifreal(X[col]))

        self.root=self.build_tree(X,y,depth=0)

    def predict(self, X: pd.DataFrame):

        predictions = X.apply(self._predict_single_sample,axis=1)

        return predictions

    def _predict_single_sample(self,sample:pd.Series):

        node=self.root
        while node.left_child or node.right_child:

            if node.node_info=='real':
                if sample[node.feature] <= node.threshhold:
                    node = node.left_child
                else:
                    node = node.right_child
            else:
                if sample[node.feature] == node.threshhold:
                    node = node.left_child
                else:
                    node = node.right_child

        return node.prediction
    
    def _get_split_values(self,column:pd.Series):
        return column.unique()

    def _calculate_split_metric(self,X,y,feature,value,discrete=None):
        if self.criterion == 'information_gain':
            return information_gain(X,y,feature,value,discrete=discrete)
        
        elif self.criterion == 'entropy':
            if discrete:
                y_left = y[X[feature]==value]
                y_right = y[X[feature]!=value]

            else:
                y_left = y[X[feature]<=value]
                y_right = y[X[feature]>value]
            return (len(y_left)/len(y))*entropy(y_left) + (len(y_right)/len(y))*entropy(y_right)
            
        else:
            if discrete:
                y_left = y[X[feature]==value]
                y_right = y[X[feature]==value]
                
            else:
                y_left = y[X[feature]<=value]
                y_right = y[X[feature]>value]
            return len(y_left)/len(y)*gini_index(y_left) + len(y_right)/len(y)*gini_index(y_right)

    
    def _is_best_split(self,curr_metric,best_metric):
        if self.criterion == 'information_gain':
            return curr_metric > best_metric
        else:
            return curr_metric < best_metric

class DecisionTreeRegressor(Node,BaseDecisionTree):

    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self, X: pd.DataFrame):
        return

@dataclass
class DecisionTree():
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5,model=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.model     = model

    def fit(self, X:pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        if not isinstance(X,pd.DataFrame):
            if not isinstance(X,np.ndarray):
                raise TypeError(f'Expected X to be a pandas Dataframe or numpy array, got {type(X)} instead')
            else:
                X=pd.DataFrame(X)
        
            
        if not isinstance(y,pd.Series):
            if not isinstance(y,np.ndarray):
                raise TypeError(f'Expected y to be a pandas series or numpy array, got {type(y)} instead')
            else:
                y=pd.Series(y)

        if X.size==0:
            raise ValueError('X is empty')
        elif y.size==0:
            raise ValueError('Y is empty')
        
        if X.shape[0]!=y.shape[0]:
            raise ValueError(f'Mismatch in the sample number of X and y, X has {X.shape[0]} samples and y has {y.shape[0]} samples')
        
        if X.isnull().any().any():
            raise ValueError('X contains Null or NaN values')

        if y.isnull().any():
            raise ValueError('y contains Null or NaN values')
        
        for col in X.columns:
            if X[col].apply(type).nunique()>=2:
                raise ValueError(f'Column {col} contains inconsistent data types')
        
        if not check_ifreal(y):
            self.model = DecisionTreeClassifier(criterion=self.criterion,max_depth=self.max_depth)
            self.model.fit(X,y)
        
        else:
            pass


    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        if not isinstance(X,pd.DataFrame):
            X=pd.DataFrame(X)
        return self.model.predict(X)


    def plot(self,node=None,depth=0) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        if node is None:
            node = self.model.root

        indent = '    '
        if node.prediction is not None:
            print(f'Prediction: {node.prediction}')
        else:
            if node.node_info == 'real':
                print(f'{indent*depth}?{node.feature} >= {node.threshhold}')
            else:
                print(f'{indent*depth}?{node.feature} == {node.threshhold}')
            
            if node.left_child is not None:
                print(f'{indent*(depth+1)}Y:',end='')
                self.plot(node.left_child,depth+1)

            if node.right_child is not None:
                print(f'{indent*(depth+1)}N:',end='')
                self.plot(node.right_child,depth+1)

# from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
y=iris.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
# tree=DecisionTreeClassifier(criterion='gini')
tree=DecisionTree(criterion='information_gain',max_depth=5)
tree.fit(x_train,y_train)
predictions=tree.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))