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
from sklearn.svm import SVR
import pickle

data = pd.read_csv('PM2_5Dataset.csv',index_col ='date',parse_dates=True)

data = data.bfill()

X=data.iloc[:,[0,2,3,4,5,6,7,8,9]].values
y=data.iloc[:,10].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25,random_state=0)

n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

def rs_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    rs =cross_val_score(model,X_train, y_train, cv = kf,scoring='r2')
    return(rs)

lr = LinearRegression()
RF = RandomForestRegressor(random_state=1, n_estimators=100)
KNN =KNeighborsRegressor(n_neighbors=2)

svr = SVR(kernel='linear')

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



stacked_averaged_models = StackingAveragedModels(base_models = (KNN,lr,RF),meta_model = svr)

stacked_averaged_models.fit(X_train, y_train)

pickle.dump(stacked_averaged_models, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[174 , 97,  84, 142, 117 ,100 ,202, 167, 166]]))

#stacked_train_pred = stacked_averaged_models.predict(X_train)
#stacked_pred = stacked_averaged_models.predict([[174 , 97,  84, 142, 117 ,100 ,202, 167, 166]])
#print(stacked_pred)




