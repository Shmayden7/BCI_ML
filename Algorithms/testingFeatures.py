import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.sparse.construct import rand, random
from sklearn.metrics import mean_squared_error
import xgboost as xgb

def testFeatures(trainingData):
    X = np.empty([row,col],int)
    y = np.empty([],int) 
    for instance in trainingData:
        holder_X = np.array(instance.ml_X)
        holder_y = np.array(instance.ml_y)

        print(holder_X)

        # X = np.append(X, holder_X, axis=0)
        # y = np.append(y, holder_y, axis=0)

    data_dmatrix = xgb.DMatrix(data=X,label=y) 
    print(data_dmatrix)
    # xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
    #             max_depth = 5, alpha = 10, n_estimators = 10)
    # xg_reg.fit(X_train,y_train)