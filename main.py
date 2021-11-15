# Imports
##################################
from os import write
from scipy.sparse import data
from Algorithms.Classes.TrainingData import TrainingData
from Algorithms.algorithms import trainMLP, trainRF, trainLDA, trainQDA, testClassifier
from Algorithms.dataSets import createDataSet, createRandomDataset
from Algorithms.Classes.Other.readWrite import readClassifier, writeClassifier, writeTrainingDataInstance, readTrainingDataInstance
##################################

# Universal Parameters
##################################
userID = 0                       # Ayden: 0, Josh: 1, Ahmad: 2
numOfTrainingFiles = 4            # Number of CSV's used to create the training dataset
numOfTestingFiles = 2             # Number of CSV's used to create the testing dataset              
nullPercentage = 0.1             # Percent of 0's used in the data set    
##################################

# LDA
##################################
divisionID_LDA = 1               # Determines which type of CSV files are read and how they're divided
featureID_LDA = 1                # Determines which features are applied to a specific division of data

def runLDA():
    trainingData = createDataSet(userID, divisionID_LDA, featureID_LDA, nullPercentage, numOfTrainingFiles)
    testingData = createRandomDataset(userID, divisionID_LDA, featureID_LDA, nullPercentage, numOfTestingFiles)

    classifierLDA = trainLDA(trainingData)
    accuracy = testClassifier(classifierLDA, testingData)

    writeClassifier(f'RF_{divisionID_LDA}_{featureID_LDA}_{accuracy}%.pkl', classifierLDA, userID)
##################################

# QDA
##################################
divisionID_QDA = 1               # Determines which type of CSV files are read and how they're divided
featureID_QDA = 1                # Determines which features are applied to a specific division of data 

def runQDA():
    trainingData = createDataSet(userID, divisionID_QDA, featureID_QDA, nullPercentage, numOfTrainingFiles)
    testingData = createRandomDataset(userID, divisionID_QDA, featureID_QDA, nullPercentage, numOfTestingFiles)

    classifierQDA = trainQDA(trainingData)
    accuracy = testClassifier(classifierQDA, testingData)

    writeClassifier(f'RF_{divisionID_QDA}_{featureID_QDA}_{accuracy}%.pkl', classifierQDA, userID)
##################################

# RF
##################################
numOfTrees = 300                 # Number of trees used in the random forest          
divisionID_RF = 1                # Determines which type of CSV files are read and how they're divided
featureID_RF = 1                 # Determines which features are applied to a specific division of data

def runRF():
    trainingData = createDataSet(userID, divisionID_RF, featureID_RF, nullPercentage, numOfTrainingFiles)
    testingData = createRandomDataset(userID, divisionID_RF, featureID_RF, nullPercentage, numOfTestingFiles)

    classifierRF = trainRF(trainingData, 200)
    accuracy = testClassifier(classifierRF, testingData)

    writeClassifier(f'RF_1_1_{accuracy}%.pkl', classifierRF, userID)
##################################

# MLP
##################################
divisionID_MLP = 1               # Determines which type of CSV files are read and how they're divided
featureID_MLP = 1                # Determines which features are applied to a specific division of data
params = {
    'hidden_layer_sizes': [(50),(100),(100),(50)],    # value is the number of neurons in a layer, length is the number of hidden layers
    # 'max_iter': 300,                                  # number of times it will run through the training data
    # 'activation': 'relu',                             # Activation function for each of the hidden layers
    # 'solver': 'adam',                                 # Activation function for the hidden layers
    # 'shuffle': True,
    # 'learning_rate': ,                              # allows you to set a seed for reproducing the same results
}

def runMLP():
    trainingData = createDataSet(userID, divisionID_MLP, featureID_MLP, nullPercentage, numOfTrainingFiles)
    testingData = createRandomDataset(userID, divisionID_RF, featureID_RF, nullPercentage, numOfTestingFiles)

    classifierMLP = trainMLP(trainingData, params)
    accuracy = testClassifier(classifierMLP, testingData)

    writeClassifier(f'MLP_{divisionID_MLP}_{featureID_MLP}_{accuracy}%.pkl', classifierMLP, userID)
##################################

# Execution Code
##################################
dataset = createDataSet(userID, 1, 1, nullPercentage, numOfTrainingFiles)
classifier = trainLDA(dataset)
testClassifier(classifier, dataset)
##################################