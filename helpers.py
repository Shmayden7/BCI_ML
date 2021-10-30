import csv
import time
import numpy as np

def sampleRateDelay(sampleRate, value):   #Delays returning the next data point from the csv file based on the sample rate of the data  
    time1 = time.time()
    while True:
        if time.time() > (time1 + 1/sampleRate):  # check to see if a tenth of a second has passed
            break

    return value

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
