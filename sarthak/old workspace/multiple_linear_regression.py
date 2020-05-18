#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:52:37 2020

@author: sarthak
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset =pd.read_csv('ALLHDPD.csv')
dataset =dataset.bfill()
dataset.info
'''
from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer= imputer.fit(X[:,[1,2,3,4,5,6,12,13,14,15,16,17,18]])
X[:,[1,2,3,4,5,6,12,13,14,15,16,17,18]]= imputer.transform(X[:,[1,2,3,4,5,6,12,13,14,15,16,17,18]])
'''

X=dataset.iloc[:,[1,3,4,5,6]].values
y=dataset.iloc[:,12].values

'''
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:, 3])    

onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

X = X[:,1:]
'''
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=.3,random_state=0)

''' not requireed already taken care
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''

#fitting multiple linear regression to dataset
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
#predicting test results
y_pred=regressor.predict(X_test)


#building optimal model using backward elimination
import statsmodels.api as sm

#adding intercept for ols_regressor or y0
X=np.append(arr=np.ones((364,1)).astype(int),values=X,axis=1)

X_opt =X[:,[0,1,2,3,4,5]] 
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt =X[:,[0,1,3,4,5]] 
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt =X[:,[0,1,4,5]] 
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()


X_opt =X[:,[0,1,5]] 
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
