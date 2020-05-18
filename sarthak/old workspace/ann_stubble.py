	# -*- coding: utf-8 -*-


# Part 1 - Data Preprocessing
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset =pd.read_csv('ALLHDPD.csv')
dataset =dataset.bfill()
X=dataset.iloc[:,[1,3,4,5,6]].values
y=dataset.iloc[:,12:13].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
regressor = Sequential()

# Adding the input layer and the first hidden layer
regressor.add(Dense(output_dim = 3,init = 'uniform', activation = 'relu', input_dim = 5))

# Adding the second hidden layer
regressor.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu'))

# Adding the output layer
regressor.add(Dense(output_dim = 1, init = 'uniform'))

# Compiling the ANN
regressor.compile( loss='mean_squared_error',optimizer = 'adam')

# Fitting the ANN to the Training set
history=regressor.fit(X_train, y_train, batch_size = 18, epochs = 364)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
#print(y_pred,y_test)


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

df1 = df.head(50)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#plt.scatter(X_test, y_test,  color='gray')
#plt.plot(X_test, y_pred, color='red', linewidth=2)
#plt.show()
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


