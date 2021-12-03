# imports 
##################################
import numpy as np
import math

from scipy.integrate import simps
from scipy.signal import welch
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

def integral(bucket): # Uses simpsons method
        sum = 0
        integralArray = []
        for col in bucket:
            for row in col:
                sum += row
            sum *= (len(col) - 1)
            integralArray.append(sum)
        
        return integralArray
##################################

# BandPower of a 2-D array over time frame, 2D by Col
##################################
def bandPower(data, band, sampleFreq, timeFrame):
    bpArray = []
    nperseg = timeFrame * sampleFreq

    bands = {
        'delta': {
            'low_f': 0.5,
            'high_f': 4
        },
        'theta': {
            'low_f': 4,
            'high_f': 8
        },
        'alpha': {
            'low_f': 8,
            'high_f': 13
        },
        'beta': {
            'low_f': 13,
            'high_f': 30
        },
        'gamma': {
            'low_f': 30,
            'high_f': 100
        }
    }

    
    f ,psd = welch(data, sampleFreq, nperseg=nperseg)

    # Frequency resolution
    fRes = f[1] - f[0]

    # Defines the band using the low/high frequency 
    band = np.logical_and(f >= bands[band]['low_f'], f <= bands[band]['high_f'])

    # Approximates band power
    bp = simps(psd[band], dx=fRes)

    bpArray.append(bp)

    #Returns bandpowers of each channel in a row vector
    return bp  # returns bandpower
##################################
