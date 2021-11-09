# Imports
##################################
from os import write
from Algorithms.algorithms import trainMLP, trainRF, trainLDA, trainQDA, testClassifier
from Algorithms.dataSets import createDataSet, createRandomDataset, readAlgorithm, writeAlgorithm
##################################

# Universal Parameters
##################################
userID = 0                       # Ayden: 0, Josh: 1, Ahmad: 2
numOfTrainingCSVs = 10           # Number of CSV's used to create the training dataset
numOfTestingCSVs = 3             # Number of CSV's used to create the testing dataset              
nullPercentage = 0.1             # Percent of 0's used in the data set    
##################################

# LDA
##################################
divisionID_LDA = 1               # Determines which type of CSV files are read and how they're divided
featureID_LDA = 1                # Determines which features are applied to a specific division of data

def runLDA():
    trainingData = createDataSet(userID, divisionID_LDA, featureID_LDA, nullPercentage, numOfTrainingCSVs)
    testingData = createRandomDataset(userID, divisionID_LDA, featureID_LDA, nullPercentage, 3)

    classifierLDA = trainLDA(trainingData)
    accuracy = testClassifier(classifierLDA, testingData)

    writeAlgorithm(f'RF_{divisionID_LDA}_{featureID_LDA}_{accuracy}%.pkl', classifierLDA, userID)
##################################

# QDA
##################################
divisionID_QDA = 1               # Determines which type of CSV files are read and how they're divided
featureID_QDA = 1                # Determines which features are applied to a specific division of data 

def runQDA():
    trainingData = createDataSet(userID, divisionID_QDA, featureID_QDA, nullPercentage, numOfTrainingCSVs)
    testingData = createRandomDataset(userID, divisionID_QDA, featureID_QDA, nullPercentage, 3)

    classifierQDA = trainQDA(trainingData)
    accuracy = testClassifier(classifierQDA, testingData)

    writeAlgorithm(f'RF_{divisionID_QDA}_{featureID_QDA}_{accuracy}%.pkl', classifierQDA, userID)
    
##################################

# RF
##################################
numOfTrees = 300                 # Number of trees used in the random forest          
divisionID_RF = 1                # Determines which type of CSV files are read and how they're divided
featureID_RF = 1                 # Determines which features are applied to a specific division of data

def runRF():
    trainingData = createDataSet(userID, divisionID_RF, featureID_RF, nullPercentage, numOfTrainingCSVs)
    testingData = createRandomDataset(userID, divisionID_RF, featureID_RF, nullPercentage, 3)

    classifierRF = trainRF(trainingData, 200)
    accuracy = testClassifier(classifierRF, testingData)

    writeAlgorithm(f'RF_1_1_{accuracy}%.pkl', classifierRF, userID)
##################################

# MLP
##################################
divisionID_MLP = 1               # Determines which type of CSV files are read and how they're divided
featureID_MLP = 1                # Determines which features are applied to a specific division of data

def runMLP():
    trainingData = createDataSet(userID, divisionID_MLP, featureID_MLP, nullPercentage, numOfTrainingCSVs)
    testingData = createRandomDataset(userID, divisionID_RF, featureID_RF, nullPercentage, 3)

    classifierMLP = trainMLP(trainingData)
    accuracy = testClassifier(classifierMLP, testingData)

    writeAlgorithm(f'MLP_{divisionID_MLP}_{featureID_MLP}_{accuracy}%.pkl', classifierMLP, userID)
##################################


# Execution Code
##################################
runMLP()
##################################