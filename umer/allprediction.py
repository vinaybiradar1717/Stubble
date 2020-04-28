import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR
from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score

from sklearn.linear_model import Lasso

from sklearn.neighbors import KNeighborsRegressor


dataset =pd.read_csv('ALLHDPD.csv')
print(dataset.columns)


dataset = dataset.drop(['Unnamed: 13',
       'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17',
       'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20'],axis=1)

dataset =dataset.bfill()
print(dataset.info())


X=dataset.iloc[:,[1,3,4,5,6]].values
y=dataset.iloc[:,12:13].values

# X = dataset['H_pm25'].values.reshape(-1,1)
# y = dataset['D_pm25'].values.reshape(-1,1)

corrMatrix = dataset.corr()
print (corrMatrix)

sns.heatmap(corrMatrix)
plt.show()



X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=.2,random_state=0)

#####################################3
# regressor = RandomForestRegressor(n_estimators = 10000, random_state = 0)
# regressor.fit(X_train, y_train)
# y_pred = regressor.predict(X_test)
###################################################

# regressor = SVR(kernel='rbf')
# regressor.fit(X_train,y_train)
# y_pred = regressor.predict(X_test)
#####################################################3

# poly_reg = PolynomialFeatures(degree = 2)
# X_poly = poly_reg.fit_transform(X_train)
# poly_reg.fit(X_poly, y_train)
# lin_reg_2 = LinearRegression()
# lin_reg_2.fit(X_poly, y_train)
# y_pred=lin_reg_2.predict(poly_reg.fit_transform(X_test))


#####################################################3

# regressor = DecisionTreeRegressor(random_state = 0)
# regressor.fit(X_train, y_train)
# y_pred = regressor.predict(X_test)
##########################################################3

# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# #X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)

# regressor = Sequential()

# # Adding the input layer and the first hidden layer
# regressor.add(Dense(output_dim = 4,init = 'uniform', activation = 'relu', input_dim = 5))

# # Adding the second hidden layer
# regressor.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu'))

# # Adding the output layer
# regressor.add(Dense(output_dim = 1, init = 'uniform'))

# # Compiling the ANN
# regressor.compile( loss='mean_squared_error',optimizer = 'adam')

# # Fitting the ANN to the Training set
# history=regressor.fit(X_train, y_train, batch_size = 1, epochs = 10)

# # Part 3 - Making the predictions and evaluating the model

# # Predicting the Test set results
# y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
# #print(y_pred,y_test)



########################################################
# est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
# # mean_squared_error(y_test, est.predict(X_test))
# y_pred = est.predict(X_test)


######################################3

reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
reg2 = RandomForestRegressor(random_state=1, n_estimators=10)

# reg3 = Lasso(alpha=1.0)
# reg4 = LinearRegression() 
reg5=KNeighborsRegressor(n_neighbors=2)
ereg = VotingRegressor(estimators=[ ('rf', reg2),('kr',reg5)])
ereg = ereg.fit(X, y)

y_pred = ereg.predict(X_test)



#######################################################33



# lasso = Lasso(alpha=1.0)
# lasso .fit(X_train, y_train)
# y_pred =lasso.predict(X_test)


#########################################################3





# regressor = LinearRegression()  
# regressor.fit(X_train, y_train)
# y_pred = regressor.predict(X_test)

#################################################333


# neigh = KNeighborsRegressor(n_neighbors=2)
# neigh.fit(X_train, y_train)
# y_pred =neigh.predict(X_test)
###################################################3
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

df1 = df.head(50)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# plt.scatter(X_test, y_test,  color='gray')
# plt.plot(X_test, y_pred, color='red', linewidth=2)
# plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('r ^2 ',r2_score(y_test, y_pred))
print('score',ereg.score(X_test, y_test))
