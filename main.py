from TrainingData import TrainingData
from Other.helpers import *

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Individual Paramaters
##################################
userID = 0     # Ayden: 0, Josh: 1, Ahmed: 2
##################################

# Creating Data Sets
##################################
def createDataSet(userID):
    set_C = TrainingData(getFileRef(0,userID))
    set_E = TrainingData(getFileRef(1,userID))
    set_F = TrainingData(getFileRef(2,userID))
    set_G = TrainingData(getFileRef(3,userID))
    set_H = TrainingData(getFileRef(4,userID))
    #set_I = TrainingData(getFileRef(5,userID))
    set_J = TrainingData(getFileRef(6,userID))
    set_K = TrainingData(getFileRef(7,userID))
    set_L = TrainingData(getFileRef(8,userID))
    set_M = TrainingData(getFileRef(9,userID))

    instanceArray = [set_C, set_E, set_F, set_G, set_H, set_I, set_J, set_K, set_L, set_M]
    return instanceArray

##################################

# ML Algorithms
##################################
def runRandomForest():
    train_instance = TrainingData(getFileRef(0,userID))
    test_instance = TrainingData(getFileRef(3,userID))

    # x_train, x_test, y_train, y_test = train_test_split(test_set.ml_X, test_set.ml_y, test_size=0.2, random_state=0)

    x_train = train_instance.ml_X
    x_test = test_instance.ml_X

    y_train = train_instance.ml_y 
    y_test = test_instance.ml_y 

    scaler = StandardScaler()
    normalized_xTrain = scaler.fit_transform(x_train)
    normalized_xTest = scaler.fit_transform(x_test)

    classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    classifier.fit(normalized_xTrain, y_train)

    y_pred = classifier.predict(normalized_xTest)

    #print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('Accuracy score:' + str(accuracy_score(y_test, y_pred)))
##################################

# Execution Code
##################################
runRandomForest()
##################################