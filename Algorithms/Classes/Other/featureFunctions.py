import numpy as np
import math

from numpy.lib.function_base import append

# Feature Functions
##################################
def mav(bucket):
    mavArray = []
    mav = 0

    for col in bucket:
        mean = np.mean(col)
        absDev = 0
        for row in col:
            absDev += abs(row - mean)
        mav = absDev/len(col)
        mavArray.append(mav)

    return mavArray
    
def aveOfCol(bucket):
    aveCol = []
    mean = 0

    for col in bucket: 
        
        mean = np.mean(col)
        aveCol.append(mean)
    
    return aveCol

def maxDiff(bucket):
    diff = []

    for col in range(len(bucket)):
        value = np.amax(bucket[col]) - np.amin(bucket[col])
        diff.append(value)
    return diff

##################################


# Random Math Functions
##################################
def var(bucket):   #Variance of columns in bucket
    varArray = []     
    for col in bucket: 

        var = 0
        mean = np.mean(bucket)

        for row in col:
            var += (row - mean)**2

        var /= (len(bucket) - 1)
        varArray.append(var)

    return varArray

def stDev(bucket):   #StDev of columns in bucket
    stDevArray = []
    varArray = var(bucket)

    for num in varArray:
        stDev = math.sqrt(num)
        stDevArray.append(stDev)

    return stDevArray

def AROC(bucket):
    slopeArray = []
    for col in bucket:
        rise = col[len(col)-1] - col[0]
        run = len(col) 
        slope = rise/run
        slopeArray.append(slope)
    
    return slopeArray

def integral(bucket): #Uses simpsons method
        sum = 0
        integralArray = []
        for col in bucket:
            for row in col:
                sum += row
            sum *= (len(col) - 1)
            integralArray.append(sum)
        
        return integralArray
##################################
