# imports 
##################################
import numpy as np
import math
import pywt

from scipy.integrate import simps
from scipy.signal import welch, coherence, lfilter 
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

# BandPower of a column over time frame, 2D by Col
##################################
def bandPower(data, band, sampleFreq, timeFrame):
    nperseg = timeFrame * sampleFreq

    bands = {
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

    #Returns bandpowers of each channel in a row vector
    return bp  # returns bandpower
##################################

# Wavelet Transform takes in a column, returns a single val
##################################
def waveletTransformProps(data):
    # cA = Low Frequency Info, cD = High Frequency Info
    cA, cD = pywt.dwt(data, 'db1')
    
    props = {
        'cA': {
            'max': np.max(cA),
            'min': np.min(cA) ,
            'mean': np.mean(cA),
            'median': np.median(cA),
            'stDev': np.std(cA), 
        },
        'cD': {
            'max': np.max(cD),
            'min': np.min(cD) ,
            'mean': np.mean(cD),
            'median': np.median(cD),
            'stDev': np.std(cD), 
        }
    }

    return props
##################################

#Sample Entropy
##################################
def SampEnt(data):
    N = len(data)
    B = 0.0
    A = 0.0
    m = 5
    r = 2
    
    # Split time series and save all templates of length m
    xmi = np.array([data[i : i + m] for i in range(N - m)])
    xmj = np.array([data[i : i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([data[i : i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn
    return -np.log(A / B)
##################################

#Hjorth Mobility
##################################
def hMob(data):
    var_y = np.var(data)
    var_dy = np.var(np.gradient(data))

    return np.sqrt(var_dy/var_y)
##################################

#Hjorth Complexity
##################################
def hCom(data):
    dy = np.gradient(data)

    return hMob(dy)/hMob(data)
##################################

#Average Power
##################################
def avgPow(bucket):
    avgPowArray = [] 

    for col in bucket:
        sum = 0
        sum = (col**2).sum()
        avgPowArray.append((sum/col.size)**0.5)

    return avgPowArray
##################################

# Auto Regression Coefficient mean
##################################
def autoRegCoeff(data):
    sum = 0 
    m = 10 #order
    yt = data[len(data) - 1]

    for i in range(m):
        if data[i*10] == 0:
            continue
        else:
            coeff = yt / data[i*10]
        
        sum += coeff

    return sum / m
