# -*- coding: utf-8 -*-

# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error

# Importing the dataset
dataset = pd.read_csv('revisedDataset.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=.25,random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = np.array(y_train).reshape(-1,1)
y_train = sc_y.fit_transform(y_train)
y_train= y_train.flatten()


# Fitting the svr to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='linear')
regressor.fit(X_train,y_train)


# Predicting a new result
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))


#results predictions
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('r ^2 ',r2_score(y_test, y_pred))
# print('score',ereg.score(y_test, y_pred))
print('explained_variance_score',explained_variance_score(y_test, y_pred))
print('max_error',max_error(y_test, y_pred))

np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))



# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [
            {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
               {'C': [1, 10, 100, 1000], 'kernel': ['linear']  }
          #   {'C': [1, 10, 100, 1000], 'kernel': ['precomputed']}
              ]
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'r2',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print(best_accuracy)
print(best_parameters)
