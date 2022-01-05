# Imports
##################################
from Algorithms.Classes.Other.utilFunctions import getPKLRef
from Algorithms.algorithms import createAndTestAlgorithm
from Algorithms.dataSets import createDataSet
from Algorithms.Classes.Other.readWrite import readClassifier, writeClassifier, writeTrainingDataInstance, readTrainingDataInstance
from Algorithms.testingFeatures import testFeatures

#testing
from Algorithms.Classes.Other.featureFunctions import waveletTransformProps

import numpy as np
##################################

# Universal Parameters
##################################
userID = 1                     # Josh: 1, Ahmad: 2, #Eryn: 3, Sejune: 4
numOfTrainingFiles = 10         # Number of CSV's used to create the training dataset
testSizePercentage = 0.3       # Number of CSV's used to create the testing dataset              
nullPercentage = 0.05          # Percent of 0's used in the data set    
##################################

# LDA
##################################
divisionID_LDA = 1               # Determines which type of CSV files are read and how they're divided

def runLDA():

    params = {

    }

    data = createDataSet(userID, divisionID_LDA, nullPercentage, numOfTrainingFiles)
    classifierLDA, accuracy = createAndTestAlgorithm(data, testSizePercentage, 'LDA', params)

    writeClassifier(f'LDA_{divisionID_LDA}_{accuracy}%.pkl', classifierLDA, userID)
##################################

# QDA
##################################
divisionID_QDA = 1               # Determines which type of CSV files are read and how they're divided

def runQDA():

    params = {

    }

    data = createDataSet(userID, divisionID_QDA, nullPercentage, numOfTrainingFiles)
    classifierQDA, accuracy = createAndTestAlgorithm(data, testSizePercentage, 'QDA', params)

    writeClassifier(f'QDA_{divisionID_QDA}_{accuracy}%.pkl', classifierQDA, userID)
##################################

# RF
##################################
divisionID_RF = 2                # Determines which type of CSV files are read and how they're divided

def runRF():

    params = {
    'numOfTrees': 300,
    'bootstrap': True,
    'max_depth': 15,
    'n_jobs' : -1,
    'random_state': 0
    }
    data = createDataSet(userID, divisionID_RF, nullPercentage, numOfTrainingFiles)   
    classifierRF, accuracy = createAndTestAlgorithm(data, testSizePercentage, 'RF', params)
    
    writeClassifier(f'RF_{divisionID_RF}_{accuracy}%.pkl', classifierRF, userID)
##################################

# MLP
##################################
divisionID_MLP = 1               # Determines which type of CSV files are read and how they're divided

def runMLP():
    
    params = {
    'hidden_layer_sizes': [(50),(100),(100),(50)],    # value is the number of neurons in a layer, length is the number of hidden layers
    # 'max_iter': 300,                                # number of times it will run through the training data
    # 'activation': 'relu',                           # Activation function for each of the hidden layers
    # 'solver': 'adam',                               # Activation function for the hidden layers
    # 'shuffle': True,
    # 'learning_rate': ,                              # allows you to set a seed for reproducing the same results
    }

    data = createDataSet(userID, divisionID_MLP, nullPercentage, numOfTrainingFiles)
 
    classifierMLP, accuracy = createAndTestAlgorithm(data, testSizePercentage, 'MLP', params)

    writeClassifier(f'MLP_{divisionID_MLP}_{accuracy}%.pkl', classifierMLP, userID)
##################################

# Execution Code
##################################
#data = createDataSet(userID, divisionID_RF, nullPercentage, numOfTrainingFiles, readPKL=False)
#writeTrainingDataInstance(data)
runRF()

################################## 