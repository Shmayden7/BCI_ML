# Imports
##################################
from Algorithms.algorithms import createAndTestAlgorithm
from Algorithms.dataSets import createDataSet
from Algorithms.Classes.Other.readWrite import readClassifier, writeClassifier, writeTrainingDataInstance, readTrainingDataInstance
from Algorithms.testingFeatures import testFeatures
##################################

# Universal Parameters
##################################
userID = 0                       # Ayden: 0, Josh: 1, Ahmad: 2
numOfTrainingFiles = 1           # Number of CSV's used to create the training dataset
testSizePercentage = 0.3         # Number of CSV's used to create the testing dataset              
nullPercentage = 0.05            # Percent of 0's used in the data set    
##################################

# LDA
##################################
divisionID_LDA = 1               # Determines which type of CSV files are read and how they're divided
featureID_LDA = 1                # Determines which features are applied to a specific division of data

def runLDA():

    params = {

    }

    data = createDataSet(userID, divisionID_LDA, featureID_LDA, nullPercentage, numOfTrainingFiles)
    classifierLDA, accuracy = createAndTestAlgorithm(data, testSizePercentage, 'LDA', params)

    writeClassifier(f'LDA_{divisionID_LDA}_{featureID_LDA}_{accuracy}%.pkl', classifierLDA, userID)
##################################

# QDA
##################################
divisionID_QDA = 1               # Determines which type of CSV files are read and how they're divided
featureID_QDA = 1                # Determines which features are applied to a specific division of data 

def runQDA():

    params = {

    }

    data = createDataSet(userID, divisionID_QDA, featureID_QDA, nullPercentage, numOfTrainingFiles)
    classifierQDA, accuracy = createAndTestAlgorithm(data, testSizePercentage, 'QDA', params)

    writeClassifier(f'QDA_{divisionID_QDA}_{featureID_QDA}_{accuracy}%.pkl', classifierQDA, userID)
##################################

# RF
##################################
divisionID_RF = 1                # Determines which type of CSV files are read and how they're divided
featureID_RF = 1                 # Determines which features are applied to a specific division of data

def runRF():

    params = {
    'numOfTrees': 200,
    'bootstrap': True,
    'max_depth': 10,
    'n_jobs' : -1,
    'random_state': 0
    }

    data = createDataSet(userID, divisionID_RF, featureID_RF, nullPercentage, numOfTrainingFiles, readPKL=False)
    classifierRF, accuracy = createAndTestAlgorithm(data, testSizePercentage, 'RF', params)

    writeClassifier(f'RF_{divisionID_RF}_{featureID_RF}_{accuracy}%.pkl', classifierRF, userID)
##################################

# MLP
##################################
divisionID_MLP = 1               # Determines which type of CSV files are read and how they're divided
featureID_MLP = 1                # Determines which features are applied to a specific division of data

def runMLP():
    
    params = {
    'hidden_layer_sizes': [(50),(100),(100),(50)],    # value is the number of neurons in a layer, length is the number of hidden layers
    # 'max_iter': 300,                                # number of times it will run through the training data
    # 'activation': 'relu',                           # Activation function for each of the hidden layers
    # 'solver': 'adam',                               # Activation function for the hidden layers
    # 'shuffle': True,
    # 'learning_rate': ,                              # allows you to set a seed for reproducing the same results
    }

    data = createDataSet(userID, divisionID_MLP, featureID_MLP, nullPercentage, numOfTrainingFiles, False)
    classifierMLP, accuracy = createAndTestAlgorithm(data, testSizePercentage, 'MLP', params)

    writeClassifier(f'MLP_{divisionID_MLP}_{featureID_MLP}_{accuracy}%.pkl', classifierMLP, userID)
##################################

# Execution Code
##################################
runRF()
##################################