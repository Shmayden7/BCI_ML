# Imports
##################################
from os import write
from Algorithms.algorithms import trainMLP, trainRF, trainLDA, trainQDA, testClassifier
from Algorithms.dataSets import createDataSet, createRandomDataset, readAlgorithm, writeAlgorithm
##################################

# User Paramaters
##################################
userID = 0                       # Ayden: 0, Josh: 1, Ahmad: 2
##################################

# Universal Parameters
##################################
numOfTrainingCSVs = 10           # Number of CSV's used to create the training dataset
numOfTestingCSVs = 3             # Number of CSV's used to create the testing dataset              
nullPercentage = 0.1             # Percent of 0's used in the data set    
##################################

# RandomForest Paramaters
##################################            
numOfTrees = 300                 # Number of trees used in the random forest          
divisionID_RF = 1                # Determines which type of CSV files are read and how they're divided
featureID_RF = 1                 # Determines which features are applied to a specific division of data
##################################

# LDA Parmeters
##################################
divisionID_LDA = 1
featureID_LDA = 1  
##################################

# LDA
##################################
def runLDA():
    trainingData = createDataSet(userID, divisionID_RF, featureID_RF, nullPercentage, numOfTrainingCSVs)
    testingData = createRandomDataset(userID, divisionID_RF, featureID_RF, nullPercentage, 3)

    classifierLDA = trainLDA(trainingData)
    accuracy = testClassifier(classifierLDA, testingData)

    writeAlgorithm(f'RF_1_1_{accuracy}%.pkl', classifierLDA, userID)
##################################

# QDA
##################################
def runQDA():
    trainingData = createDataSet(userID, divisionID_RF, featureID_RF, nullPercentage, numOfTrainingCSVs)
    testingData = createRandomDataset(userID, divisionID_RF, featureID_RF, nullPercentage, 3)

    classifierQDA = trainQDA(trainingData)
    accuracy = testClassifier(classifierQDA, testingData)

    writeAlgorithm(f'RF_1_1_{accuracy}%.pkl', classifierQDA, userID)
    
##################################

# RF
##################################
def runRF():
    trainingData = createDataSet(userID, divisionID_RF, featureID_RF, nullPercentage, numOfTrainingCSVs)
    testingData = createRandomDataset(userID, divisionID_RF, featureID_RF, nullPercentage, 3)

    classifierRF = trainRF(trainingData, 200)
    accuracy = testClassifier(classifierRF, testingData)

    writeAlgorithm(f'RF_1_1_{accuracy}%.pkl', classifierRF, userID)
##################################

# MLP
##################################
def runMLP():
    trainingData = createDataSet(userID, divisionID_RF, featureID_RF, nullPercentage, numOfTrainingCSVs)
    testingData = createRandomDataset(userID, divisionID_RF, featureID_RF, nullPercentage, 3)

    classifierMLP = trainMLP(trainingData)
    accuracy = testClassifier(classifierMLP, testingData)

    writeAlgorithm(f'MLP_1_1_{accuracy}%.pkl', classifierMLP, userID)
##################################



# Execution Code
##################################

##################################