import time
import pickle
from .dataSets import mergeInstanceData

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def trainRandomForest(instanceArray, numOfTrees):
    train_x, train_y = mergeInstanceData(instanceArray)

    tic = time.perf_counter()
    print("\nTraining The RandomForest!")
   
    scaler = StandardScaler()
    normalized_train_x = scaler.fit_transform(train_x)

    classifier = RandomForestClassifier(n_estimators=numOfTrees, random_state=0)
    classifier.fit(normalized_train_x, train_y)

    toc = time.perf_counter()
    print(f'Built RandomForest in {toc - tic:0.4}s!')

    return classifier

def testRandomForest(classifier, instanceArray):
    test_x, test_y = mergeInstanceData(instanceArray)

    tic = time.perf_counter()
    print('\nTesting the RandomForest!')

    scaler = StandardScaler()
    normalized_test_x = scaler.fit_transform(test_x)

    y_pred = classifier.predict(normalized_test_x)
    toc = time.perf_counter()

    print(f'RandomForest was built & tested in {toc - tic:0.4}s!')

    #print(confusion_matrix(y_test, y_pred))
    print(classification_report(test_y, y_pred))
    print('Accuracy score:' + str(accuracy_score(test_y, y_pred)))
    
    
