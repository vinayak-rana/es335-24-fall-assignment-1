import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
# num_average_time = 100  # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py)
def make_data(N,M,data_type):
    """
    Function to create dummy data
    """
    if data_type == 'DIDO':
        X = pd.DataFrame(np.random.randint(0,2,size=(N,M)),columns=[f'feature_{i}' for i in range(M)])
        y = pd.Series(np.random.randint(0,2,size=N),dtype='category')
    elif data_type == 'DIRO':
        X = pd.DataFrame(np.random.randint(0,2,size=(N,M)),columns=[f'feature_{i}' for i in range(M)])
        y = pd.Series(np.random.randn(N))
    
    elif data_type == 'RIDO':
        X = pd.DataFrame(np.random.randn(N,M),columns=[f'feature_{i}' for i in range(M)])
        y = pd.Series(np.random.randint(0,2,size=N),dtype='category')

    else:
        X = pd.DataFrame(np.random.randn(N,M),columns=[f'feature_{i}' for i in range(M)])
        y = pd.Series(np.random.randn(N))

    return X,y
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs

def data_split(X,y,test_size=0.2):
    """
    Function to split the dataset 
    """
    idx = int(np.ceil(0.8*len(y)))
    X_train , X_test = X[:idx+1] , X[idx+1:]
    y_train , y_test = y[:idx+1] , y[idx+1:]

    return X_train,X_test,y_train,y_test

def calculate_avg_time(tree,X_train,X_test,y_train,num_average_time):
    """
    Function to calculate the average time of a particular dataset with N entries and M features
    """
    fit_time = 0
    predict_time = 0

    for _ in range(num_average_time):
        st = time.time()
        tree.fit(X_train,y_train)
        fit_time += (time.time()-st)

        st = time.time()
        _=tree.predict(X_test)
        predict_time += (time.time()-st)

    avg_fit_time = fit_time/num_average_time
    avg_predict_time = predict_time/num_average_time

    return avg_fit_time,avg_predict_time

def experiment(values_N,values_M,num_average_time = 1,data_type = 'DIDO'):

    """
    Function to calculate average time for each value of M x N
    """
    fit_time = []
    predict_time = []

    for N in values_N:
        for M in values_M:
            print(N)
            print(M)
            X,y = make_data(N,M,data_type)
            x_train,x_test,y_train,_ = data_split(X,y)
            
            if data_type == 'DIDO' or data_type == 'RIDO':
                tree = DecisionTree(criterion='gini_index',max_depth=8)
            
            else:
                tree = DecisionTree(criterion='mse',max_depth=5)
            
            f_time , p_time = calculate_avg_time(tree,x_train,x_test,y_train,num_average_time)
            fit_time.append(f_time)
            predict_time.append(p_time)
        
    return fit_time,predict_time

def plot(N_values,M_values,times):
    """
    Function to plot the 3-D Graph
    """

    fit_time_grid = np.array(times).reshape(len(N_values), len(M_values))

    N_grid, M_grid = np.meshgrid(M_values, N_values)  

    fig = plt.figure(figsize=(11,11))
    fig = fig.add_subplot(111, projection='3d')

    fig.plot_surface(N_grid, M_grid, fit_time_grid, cmap='viridis')

    fig.set_xlabel('M')
    fig.set_ylabel('N')
    fig.set_zlabel('Processing Time')
    plt.title('N vs M vs Time taken',fontsize=20,fontweight='bold')

    plt.show()

if __name__ =='__main__':

    values_N = [10,100,1000,10000]
    values_M = [2,4,8,16,32,64,128]

    fit_time,predict_time = experiment(values_N,values_M)


    print(f'Fit time: {fit_time}')
    print(f'Predict time: {predict_time}')