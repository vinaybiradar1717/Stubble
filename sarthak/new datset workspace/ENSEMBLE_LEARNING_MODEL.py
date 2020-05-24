# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('revisedDataset.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=.20,random_state=0)





r1 = LinearRegression()
reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
reg2 = RandomForestRegressor(random_state=1, n_estimators=1000)

# reg3 = Lasso(alpha=1.0)
reg4 = LinearRegression() 
reg5=KNeighborsRegressor(n_neighbors=2)
ereg = VotingRegressor(estimators=[ ('rf', reg2),('lr', r1),('kr',reg5)])
ereg = ereg.fit(X, y)

y_pred = ereg.predict(X_test)



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



from sklearn.model_selection import cross_val_score
ereg_accuracies = cross_val_score(estimator = ereg, X = X_train, y = y_train, cv = 10,scoring='r2')
print("Mean_ereg_Acc : ", ereg_accuracies.mean())
ereg_variance=ereg_accuracies.std()
print(ereg_variance)

'''

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters={ 
            "n_estimators"      : [10,20,30,40,50,60,70,80,90,100],
            "max_features"      : ["auto", "sqrt", "log2"],
            "min_samples_split" : [2,4,8],
            "bootstrap": [True, False],
            }
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
'''