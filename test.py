import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
from math import sqrt
import lightgbm as lgb
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xg
import shap

dataset = pd.read_csv("Dataset/data.csv")
dataset.fillna(0, inplace = True)
Y = dataset['Delay'].ravel()
dataset.drop(['Delay'], axis = 1,inplace=True)
columns = dataset.columns.tolist()
'''
corr_matrix = dataset.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
if len(to_drop) > 0:
    dataset.drop(to_drop, axis=1, inplace=True)
columns = dataset.columns.tolist()
print(columns)
'''
X = dataset.values
'''
#now create and apply LIGHtGBM object on water dataset
clf = lgb.LGBMRegressor(num_leaves=100,learning_rate=0.2,boosting_type='gbdt',n_estimators=5)
clf.fit(X, Y)
#calculate importance
feature_importances = (clf.feature_importances_ / sum(clf.feature_importances_)) * 100
results = pd.DataFrame({'Features': columns, 'Importances': feature_importances})
results.sort_values(by='Importances', inplace=True)
print()
print("Temperature having less importance value so it will be removed out")
print(results)
columns = results['Features'].tolist()
importance = results['Importances'].tolist()
ax = plt.barh(results['Features'], results['Importances'])
plt.xlabel('Importance percentages')
plt.show()
'''
Y = Y.reshape(-1, 1)

sc1 = MinMaxScaler(feature_range = (0, 1))
sc2 = MinMaxScaler(feature_range = (0, 1))
X = sc1.fit_transform(X)
Y = sc2.fit_transform(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
'''
temp = []
#temp.append([np.mean(X_train[:,0]), np.mean(X_train[:,1]), np.mean(X_train[:,2]), np.mean(X_train[:,3]), np.mean(X_train[:,4]), np.mean(X_train[:,5]), np.mean(X_train[:,6])])
#temp.append([np.mean(X_test[:,0]), np.mean(X_test[:,1]), np.mean(X_test[:,2]), np.mean(X_test[:,3]), np.mean(X_test[:,4]), np.mean(X_test[:,5]), np.mean(X_test[:,6])])

temp.append(["Train", np.median(X_train[:,6])])
temp.append(["Test", np.median(X_test[:,6])])
temp.append(["Train", np.mean(X_train[:,6])])
temp.append(["Test", np.mean(X_test[:,6])])

temp = pd.DataFrame(temp, columns = ['Departure_Delay', 'Value'])
print(temp)
sns.boxplot(x='Departure_Delay',y='Value',data=temp,palette='rainbow')
plt.title("Similarity Distribution Values of Independent Features Training & Testing Data")
plt.show()
'''

#now define global variables for mse and other metrics
mse = []
rmse = []
mape = []
rsquare = []

#function to calculate MSE and other metrics
def calculateMetrics(algorithm, predict, test_labels):
    predict = predict.reshape(-1, 1)
    predict = sc2.inverse_transform(predict)
    test_label = sc2.inverse_transform(test_labels)
    predict = predict.ravel()
    test_label = test_label.ravel()
    rvalue = r2_score(test_label, predict)
    mse_value = mean_squared_error(test_label, predict)
    rmse_value = sqrt(mse_value)
    mape_value = mean_absolute_error(test_label, predict)
    mse.append(mse_value)
    rmse.append(rmse_value)
    mape.append(mape_value)
    rsquare.append(rvalue)
    print(algorithm+" MAE  : "+str(mse_value))
    print(algorithm+" RMSE : "+str(rmse_value))
    print(algorithm+" MAPE  : "+str(mape_value))
    print(algorithm+" R2  : "+str(rvalue))
    plt.plot(test_label, color = 'red', label = 'Original Delay')
    plt.plot(predict, color = 'green', label = 'Predicted Delay')
    plt.title(algorithm+' Freight Train Delay Prediction')
    plt.xlabel('Test Data')
    plt.ylabel('Predicted Freight Train Delay')
    plt.legend()
    plt.show()

'''
lgb_rg = lgb.LGBMRegressor()
lgb_rg.fit(X_train, y_train)
predict = lgb_rg.predict(X_test)
calculateMetrics("Light GBM", predict, y_test)
'''

'''
lgb_rg = RandomForestRegressor()
lgb_rg.fit(X_train, y_train.ravel())
predict = lgb_rg.predict(X_test)
calculateMetrics("Light GBM", predict, y_test)
'''

'''
lgb_rg = LinearRegression(fit_intercept=False, copy_X=False)
lgb_rg.fit(X_train, y_train.ravel())
predict = lgb_rg.predict(X_test)
calculateMetrics("Light GBM", predict, y_test)
'''
'''
lgb_rg = KNeighborsRegressor(n_neighbors=3)
lgb_rg.fit(X_train, y_train.ravel())
predict = lgb_rg.predict(X_test)
calculateMetrics("Light GBM", predict, y_test)
'''
lgb_rg = xg.XGBRegressor()
lgb_rg.fit(X_train, y_train.ravel())
predict = lgb_rg.predict(X_test)
calculateMetrics("Light GBM", predict, y_test)

explainer = shap.TreeExplainer(lgb_rg)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, feature_names=columns)


