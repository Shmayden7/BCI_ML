# Imports
##################################
from Algorithms.randomForest import runRandomForest
from Algorithms.dataSets import createDataSet
##################################

# Global Paramaters
##################################
userID = 0                       # Ayden: 0, Josh: 1, Ahmed: 2
numOfInstances = 3               # Number of CSV's used to create the data set (0=1)
nullPercentage = 0.1             # Percent of 0's used in the data set
testSize = 0.2                   # Percentage of dataset used for testing
##################################

# Execution Code
##################################
instanceArray = createDataSet(userID, numOfInstances, nullPercentage)
runRandomForest(instanceArray, testSize)
##################################