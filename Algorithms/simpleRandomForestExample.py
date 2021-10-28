import pandas as pd

#importing pratice dataset
dataset = pd.read_csv("/Users/Ayden/Desktop/bill_authentication.csv")
print(dataset.head())

#deviding set into attributes and variables
x = dataset.iloc[:,0:4].values
y = dataset.iloc[:, 4].values

# importing + initializing variables from sklearn
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# feature scaling the data using sklearn
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# defining the random forest
from sklearn.ensemble import RandomForestClassifier
# n_estimators: number of trees, random_state: ?
classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# evaluating our model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Accuracy score:' + accuracy_score(y_test, y_pred))
