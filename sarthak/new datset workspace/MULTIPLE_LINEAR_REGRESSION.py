# Multiple Linear Regression
#no  need to apply feature scaling the coefficients takes care of it


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


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
ereg_accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10,scoring='r2')
print("Mean_ereg_Acc : ", ereg_accuracies.mean())
ereg_variance=ereg_accuracies.std()
print(ereg_variance)














