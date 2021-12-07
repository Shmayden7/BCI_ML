# imports 
##################################
import numpy as np

from scipy.signal import butter, lfilter, welch
from scipy.integrate import simps
##################################


# Pre-Precessing Functions
##################################
rawData=[[],[],[],[],[],[],[],[]]        #
filteredData=[[],[],[],[],[],[],[],[]]   #

# BandPass Filters 
def bandpassFilter(data, timeFrame, lowcut, highcut, sampleFreq, order=6):
    nyq =  sampleFreq
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    for i in range(0,len(rawData)):
        rawData[i].append(data[i][0])
        if len(rawData[i]) > timeFrame*sampleFreq:
            startIndex = len(rawData[i]) - (timeFrame*sampleFreq)
            endIndex = len(rawData[i]) - 1
            filteredData[i] = lfilter(b,a ,rawData[i][startIndex:endIndex])
        else:
            startIndex = 0
            endIndex = len(rawData[i]) - 1
            filteredData[i] = lfilter(b, a, rawData[i][startIndex:endIndex])

    return filteredData # Same size as incoming filteredData

# BandPower of a 2-D array over 
def bandPower(filteredData, high_f, low_f, sampleFreq, timeFrame):
    bpArray = []
    nperseg = timeFrame * sampleFreq
    
    for col in filteredData:
        f ,psd = welch(col, sampleFreq, nperseg=nperseg)

        # Frequency resolution
        fRes = f[1] - f[0]

        # Defines the band using the low/high frequency 
        band = np.logical_and(f >= low_f, f <= high_f)

        # Approximates band power
        bp = simps(psd[band], dx=fRes)

        bpArray.append(bp)

    #Returns bandpowers of each channel in a row vector
    return bpArray  #Feature Vector

# 
def extractFeatures(filteredData):
    featureArray = []
    for col in filteredData:
        bp = bandPower(col, 8, 13 , 250, 10)
    return featureArray
##################################
