import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

def k_fold_cross_val(X,y,k,depth,criterion='gini_index'):
    accuracies = []

    fold_size = len(y)//k
    for num_fold in range(k):
        start_idx = num_fold * fold_size
        end_idx = (num_fold+1) * fold_size

        x_test = X[start_idx:end_idx]
        y_test = y[start_idx:end_idx]

        x_train = np.concatenate((X[:start_idx],X[end_idx:]))
        y_train = np.concatenate((y[:start_idx],y[end_idx:]))

        model = DecisionTree(criterion=criterion,max_depth=depth)
        model.fit(x_train,y_train)
        predictions = model.predict(x_test)

        accuracies.append(accuracy(predictions,y_test))
    return np.mean(accuracies),np.std(accuracies)

def best_decision_tree_depth(X,y):
    accuracies,stds = [],[]
    best_depth = None
    best_acc = float('-inf')
    patience = 3
    depth = 1

    while(patience):
        acc,std = k_fold_cross_val(X,y,5,depth)
        accuracies.append(acc)
        stds.append(std)

        if acc > best_acc:
            best_acc = acc
            best_depth = depth

            patience = 3
        else:
            patience -=1
        depth +=1

    return best_depth , accuracies , stds


def make_dataset():
    X, y = make_classification(
        n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)
    return X,y

def plot_data():
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

def plot_acc_std(accuracies,stds):
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.grid(True)
    plt.title('ACCURACY vs DEPTH')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.plot(accuracies,label='Accuracy',color='blue')
    plt.xlim(0,len(accuracies))

    plt.subplot(1,2,2)
    plt.grid(True)
    plt.title('Standard Deviation vs DEPTH')
    plt.xlabel('Depth')
    plt.ylabel('Standard Deviation')
    plt.plot(stds,label='Standard Deviation',color='red')
    plt.xlim(0,len(accuracies))


    plt.tight_layout()
    plt.show()

def tree_performance(X,y):
    idx = int(np.ceil(0.7*len(y)))
    X_train , X_test = X[:idx+1] , X[idx+1:]
    y_train , y_test = y[:idx+1] , y[idx+1:]

    for criteria in ['information_gain','gini_index']:
        tree = DecisionTree(criterion=criteria)
        tree.fit(X_train,y_train)
        y_hat = tree.predict(X_test)

        print("Criteria :", criteria)
        print(f"Accuracy: {accuracy(y_hat, y_test):.2f}")
        for cls in np.unique(y):
            print(f'Class {cls} - Precision: {precision(y_hat,y_test,cls):.2f}, Recall: {recall(y_hat,y_test,cls):.2f}')
        print()

if __name__ == '__main__':

    np.random.seed(42)

    X,y = make_dataset()
    plot_data()
    tree_performance(X,y)

    print('DECIDING THE BEST DEPTH OF DECISION TREE')

    best_depth , accuracies , stds = best_decision_tree_depth(X,y)
    plot_acc_std(accuracies,stds)

    print(f'Best Tree Depth = {best_depth}')