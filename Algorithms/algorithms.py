import time
from .dataSets import mergeInstanceData

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Importing Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# Training Classification Algorithms

# Random Forest Classifier
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

# Multi-layer Perceptron Classifier 
def trainMLP(instanceArray):
    train_x, train_y = mergeInstanceData(instanceArray)

    tic = time.perf_counter()
    print("\nTraining The MLP!")

    scaler = StandardScaler()
    normalized_train_x = scaler.fit_transform(train_x)

    classifier = MLPClassifier()
    classifier.fit(normalized_train_x, train_y)

    toc = time.perf_counter()
    print(f'Built MLP in {toc - tic:0.4}s!')

    return classifier

# Linear Discriminant Analysis Classifier
def trainLDA(instanceArray):
    train_x, train_y = mergeInstanceData(instanceArray)

    tic = time.perf_counter()
    print("\nTraining The LDA!")

    scaler = StandardScaler()
    normalized_train_x = scaler.fit_transform(train_x)

    classifier = LinearDiscriminantAnalysis()
    classifier.fit(normalized_train_x, train_y)

    toc = time.perf_counter()
    print(f'Built LDA in {toc - tic:0.4}s!')

    return classifier

# Quadratic Discriminant Analysis Classifier
def trainQDA(instanceArray):
    train_x, train_y = mergeInstanceData(instanceArray)

    tic = time.perf_counter()
    print("\nTraining The QDA!")

    scaler = StandardScaler()
    normalized_train_x = scaler.fit_transform(train_x)

    classifier = QuadraticDiscriminantAnalysis()
    classifier.fit(normalized_train_x, train_y)

    toc = time.perf_counter()
    print(f'Built QDA in {toc - tic:0.4}s!')

    return classifier

def testClassifier(classifier, instanceArray):
    test_x, test_y = mergeInstanceData(instanceArray)

    tic = time.perf_counter()
    print('\nTesting the Classifier!')

    scaler = StandardScaler()
    normalized_test_x = scaler.fit_transform(test_x)

    y_pred = classifier.predict(normalized_test_x)
    toc = time.perf_counter()

    print(f'Classifier was tested in {toc - tic:0.4}s!')

    #print(confusion_matrix(y_test, y_pred))
    print(classification_report(test_y, y_pred))
    print('Accuracy score:' + str(accuracy_score(test_y, y_pred)))
    
    
