# Imports
##################################
from Algorithms.algorithms import trainRandomForest, trainLDA, testClassifier
from Algorithms.dataSets import createDataSet, createRandomDataset, readAlgorithm, writeAlgorithm
##################################

# User Paramaters
##################################
userID = 0                       # Ayden: 0, Josh: 1, Ahmad: 2
##################################

# Universal Parameters
##################################
numOfTrainingCSVs = 2            # Number of CSV's used to create the training dataset
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

# Execution Code
##################################


##################################