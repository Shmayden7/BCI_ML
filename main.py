# Imports
##################################
from os import write
from scipy.sparse.construct import rand, random
from Algorithms.randomForest import trainRandomForest, testRandomForest
from Algorithms.dataSets import createDataSet, readAlgorithm, writeAlgorithm
##################################

# User Paramaters
##################################
userID = 0                       # Ayden: 0, Josh: 1, Ahmad: 2
##################################

# RandomForest Paramaters
##################################
numOfTrainingCSVs = 2            # Number of CSV's used to create the training dataset
numOfTestingCSVs = 1             # Number of CSV's used to create the testing dataset
numOfTrees = 300                 # Number of trees used in the random forest
nullPercentage = 0.1             # Percent of 0's used in the data set
divisionID_RF = 1                # Determines which type of CSV files are read and how they're divided
featureID_RF = 1                 # Determines which features are applied to a specific division of data
##################################


# Execution Code
##################################
trainingInstanceArray = createDataSet(userID, divisionID_RF, featureID_RF, nullPercentage, numOfTrainingCSVs,3)
# testingInstanceArray = createDataSet(userID, divisionID_RF, featureID_RF, nullPercentage, numOfTestingCSVs)

# classifier = trainRandomForest(trainingInstanceArray, numOfTrees)
# testRandomForest(classifier, testingInstanceArray)
##################################