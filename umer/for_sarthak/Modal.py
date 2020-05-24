import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import seaborn as sns
import pickle
# from pandas.plotting import lag_plot
from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# from sklearn.metrics import explained_variance_score
# from sklearn.metrics import max_error
# from sklearn.ensemble import VotingRegressor
# from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression 
# from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('revidedataset.csv',index_col ='date')
data = data.bfill()

X=data.iloc[:,[0,1,2,3,4,7,8,9]].values
y=data.iloc[:,10].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25,random_state=42)

lr = LinearRegression()

lr.fit(X_train, y_train)


pickle.dump(lr, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[177 ,265 ,166, 177, 145, 176, 204, 184]]))

# y_pred = ereg.predict(X_test)
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print('r ^2 ',r2_score(y_test, y_pred))
# # print('score',ereg.score(y_test, y_pred))
# print('explained_variance_score',explained_variance_score(y_test, y_pred))
# print('max_error',max_error(y_test, y_pred))
# df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# df.plot(figsize=(20,8))
# ereg_accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10,scoring='r2')
# ereg_accuracies.mean()