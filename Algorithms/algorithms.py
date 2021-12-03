import time
from .dataSets import mergeInstanceData

# Sklearn imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score

# Importing Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# Training Classification Algorithms
##################################

def createAndTestAlgorithm(instanceArray,testSizePercentage,algName,params):
    x_train, x_test, y_train, y_test = mergeInstanceData(instanceArray, testSizePercentage)

    tic = time.perf_counter()
    print("\nTraining The RandomForest!")

    # Scale Using MinMaxScaler
    scaler = MinMaxScaler()
    normalized_x_train = scaler.fit_transform(x_train)

    # Creating Classifier
    if algName == 'RF':
        classifier = RandomForestClassifier(n_estimators=params['numOfTrees'],
                bootstrap=params['bootstrap'],max_depth=params['max_depth'],
                n_jobs=params['n_jobs'],random_state=params['random_state'])
    elif algName == 'MLP':
        classifier = MLPClassifier()

    elif algName == 'LDA':
        classifier = LinearDiscriminantAnalysis()

    elif algName == 'QDA':
        classifier = QuadraticDiscriminantAnalysis()

    classifier.fit(normalized_x_train, y_train)

    toc = time.perf_counter()
    print(f'Built {algName} in {toc - tic:0.4}s!')

    accuracy = testClassifier(classifier, x_test, y_test)

    return classifier, accuracy

##################################

def testClassifier(classifier, x_test, y_test):

    tic = time.perf_counter()
    print('\nTesting the Classifier!')

    # Scale Using MinMaxScaler
    scaler = MinMaxScaler()
    normalized_x_test = scaler.fit_transform(x_test)

    y_pred = classifier.predict(normalized_x_test)
    toc = time.perf_counter()

    #accuracyScore = accuracy_score(y_test, y_pred)
    accuracyScore = classifier.score(x_test, y_test)
    percentage = int(round((accuracyScore*100),0))

    print(f'Classifier was built & tested in {toc - tic:0.4}s!')

    print(classification_report(y_test, y_pred))
    print('Accuracy score: ' + str(accuracyScore))

    return percentage
    
    
