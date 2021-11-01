import csv
import time
import numpy as np
import math
# Utility Functions
##################################
def fillBucket(bucketNumber, data):
    startingRow = bucketNumber*6
    bucket = []
    for i in range(len(data)):
        colVector = []
        for j in range(startingRow, startingRow+6):
            colVector.append(data[i][j])
        bucket.append(colVector)
    return bucket

def getFileRef(index, userID):
    localPath = [
        '/Users/Ayden/Documents/BCI/ML_Training/selectChannels/',
        'C:/Users/henrij2/Desktop/Work/Neuromore/Data/ProcessedData/',
    ]

    localFiles = [
        'C-160224.csv',
        'E-160304.csv',
        'F-160202.csv',
        'G-160412.csv',
        'H-160720.csv',
        'I-160628.csv',
        'J-161121.csv',
        'K-161108.csv',
        'L-161205.csv',
        'M-161117.csv',
    ]
    return (localPath[userID] + localFiles[index])

def sampleRateDelay(sampleRate, value):   #Delays returning the next data point from the csv file based on the sample rate of the data  
    time1 = time.time()
    while True:
        if time.time() > (time1 + 1/sampleRate):  # check to see if a tenth of a second has passed
            break

    return value

##################################


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
    var = var(bucket)

    for num in var:
        stDev = math.sqrt(num)
        stDevArray.append(stDev)

    return stDevArray

def AROC(bucket):
        rise = bucket[len(bucket)-1] - bucket[0]
        run = len(bucket) - 1
        slope = rise/run
        return slope

def integral(bucket): #Uses simpsons method
        sum = 0
        for i in range(0,len(bucket)):
            sum += bucket[i]
        
        return sum*(len(bucket)-1)
##################################
