from Classes.TrainingData import TrainingData
from localDataRefrence import getFileRef
from helpers import *

# Individual Paramaters
##################################
userID = 0     # Ayden: 0, Josh: 1, Ahmed: 2
##################################

# Creating Data Objects
##################################
def createObjects(userID):
    set_C = TrainingData(getFileRef(0,userID))
    set_E = TrainingData(getFileRef(1,userID))
    set_F = TrainingData(getFileRef(2,userID))
    set_G = TrainingData(getFileRef(3,userID))
    set_H = TrainingData(getFileRef(4,userID))
    set_I = TrainingData(getFileRef(5,userID))
    set_J = TrainingData(getFileRef(6,userID))
    set_K = TrainingData(getFileRef(7,userID))
    set_L = TrainingData(getFileRef(8,userID))
    set_M = TrainingData(getFileRef(9,userID))

    instanceArray = [set_C, set_E, set_F, set_G, set_H, set_I, set_J, set_K, set_L, set_M]
    return instanceArray
##################################

# Executing Code
##################################


##################################


