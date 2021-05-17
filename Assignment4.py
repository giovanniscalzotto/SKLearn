import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import datasets
from sklearn.cluster import KMeans

# Part 1

def Linear_Regression():
    boston_dataset = load_boston()
    boston = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)

    boston['MEDV'] = boston_dataset.target 

    X = boston.drop('MEDV', axis = 1) # define X
    Y = boston['MEDV'] # define Y

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2) # Split for Train and Test data


    lin = LinearRegression()
    lin.fit(X_train, Y_train)

    # Fit model for Training data

    y_pred = lin.predict(X_train)
    rmse = (np.sqrt(mean_squared_error(Y_train, y_pred)))
    r2 = r2_score(Y_train, y_pred)

    # Predict fitted model on Testing data

    y_test_pred = lin.predict(X_test)
    rmse_test = (np.sqrt(mean_squared_error(Y_test, y_test_pred)))
    r2_test = r2_score(Y_test, y_test_pred)

    # Summary Coefficient

    variables_name = list(boston.drop('MEDV', axis = 1))
    coefficient_values = lin.coef_
    df = pd.DataFrame({'Name' : variables_name, 'Coeff': coefficient_values})

    return (print('Train RMSE: {}'.format(rmse)),print('Train R2: {}'.format(r2)),
            print('Test RMSE: {}'.format(rmse_test)),print('Test R2: {}'.format(r2_test)),
            print(df))

Linear_Regression()

# CHAS has the most positive influence while NOX has the most negative influence.

# Part 2

def wine_dataset():

    wine = datasets.load_wine()
    X = wine.data
    Y = wine.target

    SSE = []

    for k in range(1,10):
        K_means = KMeans(n_clusters = k).fit(X)
        SSE.append(K_means.inertia_)
    
    plt.figure()
    plt.plot(range(1,10), SSE, 'bx-')
    plt.title('WINE Dataset')
    plt.xlabel('k')
    plt.ylabel("SSE")
    
    return(plt.show())

wine_dataset()

def iris_dataset():

    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    SSE = []

    for k in range(1,10):
        K_means = KMeans(n_clusters=k).fit(X)
        SSE.append(K_means.inertia_)
    
    plt.figure()
    plt.plot(range(1,10), SSE, 'bx-')
    plt.title('IRIS Dataset')
    plt.xlabel('k')
    plt.ylabel("SSE")
    
    return(plt.show())

iris_dataset()

# For both dataset I confirm that 3 is the correct number of groups because there is a huge drop until it.
# After 3, the total distance changes slowly and also the slope does not decrease fastly, 
# meaning that it will not contribute to the algorithm. 







