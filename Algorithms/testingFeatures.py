import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.sparse.construct import rand, random
from sklearn.metrics import mean_squared_error
import xgboost as xgb

from Algorithms.Classes.Other.readWrite import readClassifier

def testFeatures(fileName, userID):
  
    classifier = readClassifier(fileName, userID)
    importance = classifier.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()
