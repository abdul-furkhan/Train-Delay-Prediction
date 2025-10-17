#importing pythom classes and packages
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
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
from sklearn.metrics import mean_absolute_percentage_error


main = tkinter.Tk()
main.title("Rail Delay") #designing main screen
main.geometry("1300x1200")

global filename, dataset, X_train, X_test, y_train, y_test, X, Y, sc1,sc2, pca
global mae, rmse, rsquare, mape, values,cnn_model,temp,xgboost
mae = []
rmse = []
mape = []
rsquare = []
temp = []

def uploadDataset():
    global filename, dataset, labels, values,X, Y
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,'Dataset loaded\n\n')
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True) #replace missing values with 0
    text.insert(END,str(dataset))
    
def graphLightGBM():
    global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca
    global mae, rmse, rsquare, mape, values,cnn_model
    X = dataset.values
    clf = lgb.LGBMRegressor(num_leaves=100,learning_rate=0.2,boosting_type='gbdt',n_estimators=5)
    clf.fit(X) #train LGBM on X and Y training data
    #calculate importance
    feature_importances = (clf.feature_importances_ / sum(clf.feature_importances_)) * 100
    results = pd.DataFrame({'Features': columns, 'Importances': feature_importances})
    results.sort_values(by='Importances', inplace=True)
    text.insert(END,"Features with importance values"+"\n")
    text.insert(END,results+"\n")
    columns = results['Features'].tolist()
    importance = results['Importances'].tolist()
    ax = plt.bar(results['Features'], results['Importances'])
    plt.xlabel('Importance percentages')
    plt.xticks(rotation = 90)
    plt.show()

    
def processDataset():
    global dataset, X, Y
    global X_train, X_test, y_train, y_test, pca, sc1,sc2,temp
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    data = dataset.values
    X = data[:,1:data.shape[1]-1]
    Y = data[:,data.shape[1]-1]
    Y = Y.astype(int)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)#shuffle dataset values
    X = X[indices]
    Y = Y[indices]

    Y = Y.reshape(-1, 1)
    sc1 = MinMaxScaler(feature_range = (0, 1))
    sc2 = MinMaxScaler(feature_range = (0, 1))
    X = sc1.fit_transform(X) #normalize features
    Y = sc2.fit_transform(Y)
    text.insert(END,"Normalized Features\n")
    text.insert(END,X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    text.insert(END,"\n\nDataset Train & Test Split Details\n")
    text.insert(END,"80% dataset for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset for testing  : "+str(X_test.shape[0])+"\n")

    temp = []
    temp.append(["Train", np.median(X_train[:,6])])
    temp.append(["Test", np.median(X_test[:,6])])
    temp.append(["Train", np.mean(X_train[:,6])])
    temp.append(["Test", np.mean(X_test[:,6])])
    temp = pd.DataFrame(temp, columns = ['Departure_Delay', 'Value'])
    sns.boxplot(x='Departure_Delay',y='Value',data=temp,palette='rainbow')
    plt.title("Similarity Distribution Values of Independent Features Training & Testing Data")
    plt.show()
    


def calculateMetrics(algorithm, predict, test_labels):
    global sc1,sc2
    predict = predict.reshape(-1, 1)
    mape_value = mean_absolute_percentage_error(test_labels, predict)   
    predict = sc2.inverse_transform(predict)
    test_label = sc2.inverse_transform(test_labels)
    predict = predict.ravel()
    test_label = test_label.ravel()
    rvalue = r2_score(test_label, predict)
    mse_value = mean_squared_error(test_label, predict)
    rmse_value = sqrt(mse_value)
    mae_value = mean_absolute_error(test_label, predict)
    mae.append(mae_value)
    rmse.append(rmse_value)
    mape.append(mape_value)
    rsquare.append(rvalue)
    text.insert(END,algorithm+" MAE  : "+str(mae_value)+"\n")
    text.insert(END,algorithm+" RMSE : "+str(rmse_value)+"\n")
    text.insert(END,algorithm+" MAPE  : "+str(mape_value)+"\n")
    text.insert(END,algorithm+" R2  : "+str(rvalue))
    plt.plot(test_label, color = 'red', label = 'Original Delay')
    plt.plot(predict, color = 'green', label = 'Predicted Delay')
    plt.title(algorithm+' Freight Train Delay Prediction')
    plt.xlabel('Test Data')
    plt.ylabel('Predicted Freight Train Delay')
    plt.legend()
    plt.show()

def runLGBM():
    global X_train, y_train, X_test, y_test
    global predict
    text.delete('1.0', END)
    
    lgb_rg = lgb.LGBMRegressor()
    lgb_rg.fit(X_train, y_train.ravel())#train the model
    predict = lgb_rg.predict(X_test)#perform prediction on test data
    calculateMetrics("Light GBM", predict, y_test)#calculate metrics using original and predicted labels

def runLinearRegression():
    global X_train, y_train, X_test, y_test
    global predict
    text.delete('1.0', END)
   
    lr = LinearRegression(fit_intercept=False, copy_X=False)
    lr.fit(X_train, y_train.ravel())#train the model
    predict = lr.predict(X_test)#perform prediction on test data
    calculateMetrics("Linear Regression", predict, y_test)#calculate metrics using original and predicted labels

def runKNN():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    
    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(X_train, y_train.ravel())#train the model
    predict = knn.predict(X_test)#perform prediction on test data
    calculateMetrics("KNN", predict, y_test)#calculate metrics using original and predicted labels

def runRF():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train.ravel())#train the model
    predict = rf.predict(X_test)#perform prediction on test data
    calculateMetrics("Random Forest", predict, y_test)#calculate metrics using original and predicted labels

def runXGBoost():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore,xgboost
    text.delete('1.0', END)
    
    xgboost = xg.XGBRegressor()
    xgboost.fit(X_train, y_train.ravel())#train the model
    predict = xgboost.predict(X_test)#perform prediction on test data
    calculateMetrics("Extension XGBoost", predict, y_test)#calculate metrics using original and predicted labels


def comparisongraph():
    df = pd.DataFrame([['Light GBM','R2',rsquare[0]],['Light GBM','RMSE',rmse[0]],
                       ['Linear Regression','R2',rsquare[1]],['Linear Regression','RMSE',rmse[1]],
                       ['Random Forest','R2',rsquare[2]],['Random Forest','RMSE',rmse[2]],
                       ['KNN','R2',rsquare[3]],['KNN','RMSE',rmse[3]],
                       ['Extension XGBoost','R2',rsquare[4]],['Extension XGBoost','RMSE',rmse[4]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.title("All Algorithms Performance Graph")
    plt.show()

def prdeict():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore,cnn_model
    global sc1,sc2,xgboost
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True) #remove missing values
    temp = dataset.values
    testData = dataset.values
    #normalizing test data
    testData = sc1.transform(testData)
    #perform prediction on test
    predict = xgboost.predict(testData)
    #reshaping and denormalize predicted output
    predict = predict.reshape(-1, 1)
    predict = sc2.inverse_transform(predict)
    for i in range(len(predict)):#now loop each test data and then predict traffic volume
        text.insert(END,"Test Data = "+str(temp[i])+"\n")
        text.insert(END,"Predicted Delay ====> "+str(predict[i,0])+"\n")
        print()

font = ('times', 16, 'bold')
title = Label(main, text='Short-Term Arrival Delay Time Prediction in Freight Rail Operations Using Data-Driven Models')
title.config(bg='white',fg='brown1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=27,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Attack Database", command=uploadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Preprocess & Split Dataset", command=processDataset)
processButton.place(x=250,y=100)
processButton.config(font=font1)

lgbmButton = Button(main, text="Run LightGBM Algorithm", command=runLGBM)
lgbmButton.place(x=490,y=100)
lgbmButton.config(font=font1)

lrButton = Button(main, text="Run LinearRegression", command=runLinearRegression)
lrButton.place(x=730,y=100)
lrButton.config(font=font1)

knnButton = Button(main, text="Run KNN Algorithm", command=runKNN)
knnButton.place(x=970,y=100)
knnButton.config(font=font1)

rfButton = Button(main, text="Run Random Forest", command=runRF)
rfButton.place(x=1200,y=100)
rfButton.config(font=font1)


xgButton = Button(main, text="Run XGBoost Algorithm", command=runXGBoost)
xgButton.place(x=10,y=150)
xgButton.config(font=font1)


graphButton = Button(main, text="Comparison Graph", command=comparisongraph)
graphButton.place(x=250,y=150)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Attack from Test Data", command=prdeict)
predictButton.place(x=490,y=150)
predictButton.config(font=font1)

main.config(bg='orange')
main.mainloop()
