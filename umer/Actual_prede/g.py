import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('PM2_5Dataset.csv',index_col='date',parse_dates=True)
# data.plot()
data['Week']=data.index.week
data['Month']=data.index.month
data['Year']=data.index.year
l = ['panipat pm 2.5','sirsa pm 2.5','Bhiwanipm 2.5','rohtak pm 2.5','patiala pm 2.5','ludhiana pm 2.5','kaithal pm 2.5','karnal pm 2.5','hisarpm25','jindpm25']
#for i in l:
    #monthwise=data.groupby(by='Year')[i,'delhi_wazirpm2.5']

    #monthwise.plot(figsize=(18,5),kind='line')
    #plt.show()
   	
#LinearRegression
#RandomForestRegressor
#SVR
#GradientBoostingRegressor
#KNeighborsRegressor
#stacked_regression

l = ['panipat pm 2.5','sirsa pm 2.5','Bhiwanipm 2.5','rohtak pm 2.5','patiala pm 2.5','ludhiana pm 2.5','kaithal pm 2.5','karnal pm 2.5','hisarpm25','jindpm25']
# x = data['panipat pm 2.5']
y = data['delhi_wazirpm2.5']
#for i in l:
    #x = data[i]
    #plt.scatter(x,y)
    #plt.xlabel(i+'pm2.5')
    #plt.ylabel('Delhi pm2.5')
    #plt.show()

r2 = pd.DataFrame({'LinearRegression_r2':[0.7731],'stacked_regression_r2':[0.7731],'lasso_r2':[0.7731],'SVR_r2':[0.7622],'xgb_r2':[0.8514],'GradientBoostingRegressor_r2':[0.8258 ],'KNeighborsRegressor_r2':[0.8313],'RandomForestRegressor_r2':[0.8111]
})
#r2.plot(kind='bar')
#plt.show()


RMSE = pd.DataFrame({'LinearRegression_RMSE':[42.8770], 
                     
                    'RandomForestRegressor_RMSE':[38.8944],
                       
                      'KNeighborsRegressor_RMSE':[36.2265],
                      'GradientBoostingRegressor_RMSE':[38.7571],
                      'XGB_RMSE':[34.5043],'SVR_RMSE':[44.7869],'Lasso_RMSE':[42.8766],'stacked_regression_RMSE':[34.3906]})
#RMSE.plot(kind ='bar')
#plt.show()


SAM = pd.DataFrame({'actual':[161. ,154., 248., 145. ,441., 272., 146., 105., 217. ,135., 194., 191., 162., 109.
, 142., 122., 728.,  98. ,257., 151., 137., 143., 420., 134.,  59., 157., 115., 114.,
 112., 446., 204.,  89.,  69., 135., 207. ,146., 111., 119., 334. ,138., 104. ,134.,
 326., 215., 104., 237., 160. ,122., 188., 106.],
                   'predection':[151.52615512 ,155.53719235, 229.98026479 ,116.74844743, 389.22391645 ,343.70933596, 149.25885975 , 93.94763365 ,199.3071945 , 108.68393756 ,186.08132659 ,182.52548311 ,158.36392282 ,106.04676157, 134.00417257, 124.69812057, 473.48595941 , 96.8218122 , 204.04468663 ,120.43748157 ,131.78383661 ,129.79577951, 419.76176857, 158.04577674 , 75.06022299, 134.76458645, 106.15482589, 151.46645547, 136.64103995 ,395.60414243, 225.45590217 ,116.53493803 , 80.34976997, 130.23137633 ,218.41083355 ,153.13752099,  97.40236473, 128.54481349, 327.76384571 ,108.38272144, 106.18855259 ,120.50358775 ,322.58145817, 208.72016145, 133.91844428 ,236.68713505, 157.78520175 ,132.1145425 , 160.73230198 ,146.74903436]
                   })
#SAM.plot(figsize=(20,8))
#plt.show()

KNN_A_P = pd.read_csv('KNN_A_P.csv')
KNN_A_P.plot()
plt.show()

RF_A_P = pd.read_csv('RF_A_P.csv')
RF_A_P.plot()
plt.show()

XGB_A_P = pd.read_csv('XGB_A_P.csv')
XGB_A_P.plot()
plt.show()
