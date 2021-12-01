import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import dask.dataframe as dd
import pickle

temp = dd.read_csv('/Users/Ayden/Documents/BCI/ML_Training/set_1_9/1.csv', dtype=float)
temp = temp.astype(float)
columns = []
for i in range(9):
  columns.append('F_'+str(i))

#print(temp.head())

df = pd.DataFrame(np.array(temp), columns = columns)
y = df.pop('F_8')
X = df

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

# for x in range(3):
#   print(X[x])
#   print(y[x])
  
min_max_scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train = min_max_scaler.transform(X_train)
X_test = min_max_scaler.transform(X_test)

#clf = xgb.XGBClassifier(tree_method = 'hist',max_depth = 10, n_estimators=200, n_jobs = -1)
clf = RandomForestClassifier(max_depth = 10, n_estimators=200,n_jobs = -1)

print("Start Training...")

clf.fit(X_train,y_train)

print("Training Complete")

print(clf.score(X_test,y_test))
