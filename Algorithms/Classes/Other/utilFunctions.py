import time

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
        # 'test_1.csv',
        # 'test_2.csv',
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
