import time

# Datasets
##################################
fileName = [
    # Josh
    'A-160308',
    'A-160310',
    'B-160225',
    'B-160229',
    'C-160224',
    'C-160302',
    'E-160219',
    'E-160226',
    # Eryn
    'F-160203',
    'F-160204',
    'G-160301',
    'G-160322',
    'H-160722',
    'I-160609',
    #Sejune
    'K-161027',
    'L-161116',
    'M-161121',
    'M-161124',
    'B-160218',
    'E-160304',
    # New Data set
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
def fillBucket(bucketNum, data):
    startingRow = bucketNum*6
    bucket = []

    for i in range(len(data)):
        colVector = []
        for j in range(startingRow, startingRow+6):
            colVector.append(data[i][j])
        bucket.append(colVector)

    return bucket

def getCSVRef(index, userID, divisionID):
    localPath = [
        f'/Users/Ayden/Documents/BCI/ML_Training/{sets[divisionID]}/',
        'C:/Users/henrij2/Desktop/Work/Data/rawDataWithMarkers/',
        ' ',
        #Eryn,
        #Sejune
    ]

    return (localPath[userID] + fileName[index] + '.csv')

def getPKLRef(index, userID, divisionID):
    localPath = [
        f'/Users/Ayden/Documents/BCI/ML_Training/{sets[divisionID]}/',
        'C:/Users/henrij2/Desktop/Work/Data/lessDataPklFiles/',
        ' ',
        #Eryn,
        #Sejune
    ]

    return (localPath[userID] + fileName[index] + '.pkl')

def getClassifierRef(userID):
    localPath = [
        '/Users/Ayden/Documents/BCI/ML_Training/classifiers/',
        'C:/Users/henrij2/Desktop/Work/Classifiers/'       
    ]

    return localPath[userID]
