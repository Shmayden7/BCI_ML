# imports 
##################################
from scipy.signal import butter, lfilter, welch
import pywt
##################################


# Pre-Precessing Helper Functions
##################################
rawData=[[],[],[],[],[],[],[],[]]        #
filteredData=[[],[],[],[],[],[],[],[]]   #

# BandPass Filters 
def bandpassFilter(data, lowcut=8, highcut=30, timeFrame = 30, sampleFreq=250, liveData=True, order=5):
    nyq =  sampleFreq
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    if liveData == True:
        # Incoming Data: [[#],[#],[#],[#],[#],[#],[#],[#]]
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
    
    else:
        filteredData = lfilter(b,a ,data)

    return filteredData # Same size as incoming filteredData

def waveletTransformLive(filteredData):
    coefficientArray = []
    for col in filteredData:
        cA, cD = pywt.dwt(col, 'db1')
        coefficientArray.append(cA + cD)

    return coefficientArray

##################################

# Feature Extraction Function
##################################
#def determineFeatureRow(filteredData,):

##################################