from numpy import true_divide
import sklearn as sk

from ...helpers import *
from ..Classes.TrainingData import TrainingData
from ..localDataRefrence import getFileRef

x = [[],[]]
y = []

# X = instance.data[range(0,24),:].transpose()
# y = data[24,:]

def setMl_X(tdInstance):
    # Filling x (feature) matrix
    # F1: mav of electrodes F2: F3:

    #  Sudo Code
    ##################################
    
    # 1: create a bucket from data prop
    # 2: calculate mav for channels []
    # 3: calculate avevoltage for channels []
    # 4: append 

    ##################################
    x = []
    # Number of full buckets in instance
    numOfBuckets = tdInstance.data // 6
    for bucketNum in range(1):
        currentCol = []
        currentBucket = fillBucket(bucketNum, tdInstance.data) # Bucketing Dat
        
        channels = currentBucket[range(0,8),:]
        print(channels)

        #mavArr = mav(currentBucket[range(0,8),:]) # array mav's for current bucket
        
        

ins_1 = TrainingData(0, getFileRef(9)) 
setMl_X(ins_1)
