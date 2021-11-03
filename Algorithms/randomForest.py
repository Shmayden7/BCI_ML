from .dataSets import mergeInstanceData

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def runRandomForest(instanceArray, testSize=0.2):

    x, y = mergeInstanceData(instanceArray)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testSize, random_state=0)

    print("\nTraining The RandomForest!")
    scaler = StandardScaler()
    normalized_xTrain = scaler.fit_transform(x_train)
    normalized_xTest = scaler.fit_transform(x_test)

    classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    classifier.fit(normalized_xTrain, y_train)

    print('\nTesting the Random Forest!')
    y_pred = classifier.predict(normalized_xTest)

    #print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('Accuracy score:' + str(accuracy_score(y_test, y_pred)))