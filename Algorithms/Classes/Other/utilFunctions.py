import time

# Datasets
##################################
fileName = [
    'C-160224',
    'E-160304',
    'F-160202',
    'G-160412',
    'H-160720',
    'I-160628',
    'J-161121',
    'K-161108',
    'L-161205',
    'M-161117',
]

sets = {
    1: 'set_1_25',
    2: 'set_1_9'
}
##################################

# Functions
def fillBucket(bucketNumber, eeg, fft):
    startingRow = bucketNumber*6
    bucket = []

    for i in range(len(fft)):
        colVector = []
        for j in range(startingRow, startingRow+6):
            colVector.append(fft[i][j])
        bucket.append(colVector)

    for i in range(len(eeg)):
        colVector = []
        for j in range(startingRow, startingRow+6):
            colVector.append(eeg[i][j])
        bucket.append(colVector)

    print(len(bucket))
    return bucket

def getCSVRef(index, userID, divisionID):
    localPath = [
        f'/Users/Ayden/Documents/BCI/ML_Training/{sets[divisionID]}/',
        'C:/Users/henrij2/Desktop/Work/Data/ProcessedData/',
    ]

    return (localPath[userID] + fileName[index] + '.csv')

def getPKLRef(index, userID, divisionID):
    localPath = [
        f'/Users/Ayden/Documents/BCI/ML_Training/{sets[divisionID]}/',
    ]

    return (localPath[userID] + fileName[index] + '.pkl')

def getClassifierRef(userID):
    localPath = [
        '/Users/Ayden/Documents/BCI/ML_Training/classifiers/',
        'C:/Users/henrij2/Desktop/Work/Classifiers/'
    ]

    return localPath[userID]

def sampleRateDelay(sampleRate, value):   #Delays returning the next data point from the csv file based on the sample rate of the data  
    time1 = time.time()
    while True:
        if time.time() > (time1 + 1/sampleRate):  # check to see if a tenth of a second has passed
            break

    return value