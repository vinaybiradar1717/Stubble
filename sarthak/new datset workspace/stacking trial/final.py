# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from scipy import stats
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression 

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


'''
#Now let's import and put the train and test datasets in  pandas dataframe
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
dataset = pd.read_csv('revisedDataset.csv')
X_train = train.iloc[:, 1:-1].values
y_train = train.iloc[:, -1].values
X_test  =  test.iloc[:, 1:-1].values
y_test  =  test.iloc[:, -1].values
#ntrain = train.shape[0]
#ntest = test.shape[0]
'''


#Now let's import and put the train and test datasets in  pandas dataframe

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


##display the first five rows of the train dataset.
train.head(5)


#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))



#Save the 'Id' column
train_date = train['date']
test_date = test['date']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("date", axis = 1, inplace = True)
test.drop("date", axis = 1, inplace = True)

#Validation function
n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values , y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


'''
from sklearn.model_selection import cross_val_score
ereg_accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10,scoring='r2')
print("Mean_ereg_Acc : ", ereg_accuracies.mean())
ereg_variance=ereg_accuracies.std()
print(ereg_variance)

'''

def rs_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rs = cross_val_score(estimator=model,train.values, y_train, cv = kf,scoring='r2')
    return(rs)


ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.delhi_wazirpm.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['delhi_wazirpm'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))



train = all_data[:ntrain]
test = all_data[ntrain:]


#BASE MODELS
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))



ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))



KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)



GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)



model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)



model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

lr = LinearRegression()

RF = RandomForestRegressor(random_state=1, n_estimators=100)


KNN =KNeighborsRegressor(n_neighbors=2)


#BASE MODEL SCORES

score = rmsle_cv(lasso)
rsscore = rs_cv(lasso)
print(score)
print("\n Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
print(rsscore)
print("\n Lasso score: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))
print("\n\n===============================")

score = rmsle_cv(ENet)
rsscore = rs_cv(ENet)
print(score)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
print(rsscore)
print("\n ElasticNet score: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))
print("\n\n===============================")

score = rmsle_cv(KRR)
rsscore = rs_cv(KRR)
print(score)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
print(rsscore)
print("\nLR score: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))
print("\n\n")

score = rmsle_cv(GBoost)
rsscore = rs_cv(GBoost)
print(score)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
print(rsscore)
print("\n GBoostscore: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))
print("\n\n===============================")

score = rmsle_cv(model_xgb)
rsscore = rs_cv(model_xgb)
print(score)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
print(rsscore)
print("\n Xgboost score: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))
print("\n\n===============================")



score = rmsle_cv(lr)
rsscore = rs_cv(lr)
print(score)
print("\n LR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
print(rsscore)
print("\n LR score: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))
print("\n\n===============================")


score = rmsle_cv(RF)
rsscore = rs_cv(RF)
print(score)
print("\n random forest score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
print(rsscore)
print("\n random forest score: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))
print("\n\n===============================")

score = rmsle_cv(KNN)
rsscore = rs_cv(KNN)
print(score)
print("\n KNN score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
print(rsscore)
print("\n KNN score: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))
print("\n\n===============================")































