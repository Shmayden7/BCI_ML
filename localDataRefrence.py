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
        'I-1606286.csv',
        'J-1611216.csv',
        'K-1611086.csv',
        'L-1612056.csv',
        'M-1611176.csv',
    ]
    return (localPath[userID] + localFiles[index])