#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression 
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn import metrics


# In[7]:


data = pd.read_csv('PM2_5Dataset.csv',index_col ='date',parse_dates=True)

data = data.bfill()

X=data.iloc[:,[0,2,3,4,5,6,7,8,9]].values
y=data.iloc[:,10].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25,random_state=0)


# In[8]:


n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

def rs_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    rs =cross_val_score(model,X_train, y_train, cv = kf,scoring='r2')
    return(rs)


# In[9]:


lr = LinearRegression()
score = rmsle_cv(lr)
rsscore = rs_cv(lr)
# print(score)
print("\nLR RMSE score of training data : {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# print(rsscore)
print("\nLR R2 score of training data: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))


# In[10]:


#Random forest
RF = RandomForestRegressor(random_state=1, n_estimators=100)
score = rmsle_cv(RF)
rsscore = rs_cv(RF)
print(score)
print("\nRF RMSE score of training data : {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

print(rsscore)
print("\nRF R2 score of training data: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))


# In[12]:


KNN =KNeighborsRegressor(n_neighbors=2)
score = rmsle_cv(KNN)
rsscore = rs_cv(KNN)
print(score)
print("\nKNN RMSE score of training data: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

print(rsscore)
print("\nKNN R2 score of training data: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))


# In[13]:


GBoost = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
score = rmsle_cv(GBoost)
rsscore = rs_cv(GBoost)
print(score)
print("\n GBoost RMSE score of training data: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

print(rsscore)
print("\n GBoost R2 score of training data: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))


# In[15]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
score = rmsle_cv(model_xgb)
rsscore = rs_cv(model_xgb)
print(score)
print("\n model_xgb RMSE score of training data: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

print(rsscore)
print("\n model_xgb R2 score of training data: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))


# In[16]:


from sklearn.svm import SVR
svr = SVR(kernel='linear')
score = rmsle_cv(svr)
rsscore = rs_cv(svr)
print(score)
print("\n svr RMSE score of training data: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

print(rsscore)
print("\n svr R2 score of training data:: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))


# In[17]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmsle_cv(lasso)
rsscore = rs_cv(lasso)
print(score)
print("\nLasso RMSE score of training data: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

print(rsscore)
print("\nLasso R2 score of training data: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))


# In[18]:


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
#         print(self.base_models)
        self.meta_model = meta_model
#         print(self.meta_model)
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
#         print(self.base_models)
        self.base_models_ = [list() for x in self.base_models]
#         print(self.base_models_)
        self.meta_model_ = clone(self.meta_model)
#         print(self.meta_model_)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
#         print(kfold)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
#         print(out_of_fold_predictions.shape)
        for i, model in enumerate(self.base_models):
#             print(i,model)
            for train_index, holdout_index in kfold.split(X,y):
#                 print(len(train_index), len(holdout_index))
                instance = clone(model)
#                 print(instance)
                self.base_models_[i].append(instance)
#                 print(self.base_models_[i])
#                 print(X[train_index], y[train_index])
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
#                 print(y_pred,y[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
#                 print(out_of_fold_predictions.shape)
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
#         print(self.meta_model_)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        print(meta_features.shape)
        return self.meta_model_.predict(meta_features)


# In[19]:


stacked_averaged_models = StackingAveragedModels(base_models = (KNN,lr,RF),meta_model = svr)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models RMSE score of training data: {:.4f} ({:.4f})".format(score.mean() , score.std()))
rsscore = rs_cv(stacked_averaged_models)
print("\nStaking r2 R2 score of training data: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))


# In[20]:



#testing 

stacked_averaged_models.fit(X_train, y_train)
stacked_train_pred = stacked_averaged_models.predict(X_train)
stacked_pred = stacked_averaged_models.predict(X_test)
# print(y_train,stacked_pred)
df = pd.DataFrame({'Actual':y_test, 'Predicted': stacked_pred})
print(df)
df.plot(figsize=(20,8))


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, stacked_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, stacked_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, stacked_pred)))
print('r ^2 ',r2_score(y_test, stacked_pred))
# print('score',ereg.score(y_test, y_pred))
print('explained_variance_score',explained_variance_score(y_test, stacked_pred))
print('max_error',max_error(y_test, stacked_pred))


# In[22]:


model_xgb.fit(X_train, y_train)
xgb_train_pred = model_xgb.predict(X_test)
# print(len(xgb_train_pred),len(y_test))
df = pd.DataFrame({'Actual':y_test, 'Predicted': xgb_train_pred})
print(df)
df.plot(figsize=(20,8))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, xgb_train_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, xgb_train_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, xgb_train_pred)))
print('r ^2 ',r2_score(y_test, xgb_train_pred))
# print('score',ereg.score(y_test, y_pred))
print('explained_variance_score',explained_variance_score(y_test, xgb_train_pred))
print('max_error',max_error(y_test, xgb_train_pred))


# In[23]:


KNN =KNeighborsRegressor(n_neighbors=2)
KNN.fit(X_train, y_train)
KNN_predict = KNN.predict(X_test)
df = pd.DataFrame({'Actual':y_test, 'Predicted': KNN_predict})
print(df)
df.plot(figsize=(20,8))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, KNN_predict))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, KNN_predict))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, KNN_predict)))
print('r ^2 ',r2_score(y_test, KNN_predict))
# print('score',ereg.score(y_test, y_pred))
print('explained_variance_score',explained_variance_score(y_test, KNN_predict))
print('max_error',max_error(y_test, KNN_predict))


# In[24]:


RF = RandomForestRegressor(random_state=1, n_estimators=100)
RF.fit(X_train, y_train)
RF_predict = RF.predict(X_test)
df = pd.DataFrame({'Actual':y_test, 'Predicted': RF_predict})
print(df)
df.plot(figsize=(20,8))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, RF_predict))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, RF_predict))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, RF_predict)))
print('r ^2 ',r2_score(y_test, RF_predict))
# print('score',ereg.score(y_test, y_pred))
print('explained_variance_score',explained_variance_score(y_test, RF_predict))
print('max_error',max_error(y_test, RF_predict))


# In[ ]:




