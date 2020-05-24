import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# %matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics


pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
# import lightgbm as lgb
data = pd.read_csv('revisedDataset.csv',index_col ='date',parse_dates=True)
data = data.bfill()
X=data.iloc[:,[0,1,2,3,4,7,8,9]].values
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

from sklearn.linear_model import LinearRegression 

lr = LinearRegression()
score = rmsle_cv(lr)
rsscore = rs_cv(lr)
print(score)
print("\nLR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

print(rsscore)
print("\nLR score: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))

#Random forest
RF = RandomForestRegressor(random_state=1, n_estimators=10)
score = rmsle_cv(RF)
rsscore = rs_cv(RF)
print(score)
print("\nLR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

print(rsscore)
print("\nLR score: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))

from sklearn.neighbors import KNeighborsRegressor

KNN =KNeighborsRegressor(n_neighbors=2)
score = rmsle_cv(RF)
rsscore = rs_cv(RF)
print(score)
print("\nLR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

print(rsscore)
print("\nLR score: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmsle_cv(RF)
rsscore = rs_cv(RF)
print(score)
print("\nLR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

print(rsscore)
print("\nLR score: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
stacked_averaged_models = StackingAveragedModels(base_models = (RF, lr, KNN),meta_model = lasso)
score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
print(rsscore)
print("\nLR score: {:.4f} ({:.4f})\n".format(rsscore.mean(), rsscore.std()))