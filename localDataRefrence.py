def getChannelValue(index):
    channels = [
        'Fp1','Fp2','F3','F4',
        'C3','C4','P3','P4',
        'O1','O2','A1','A2',
        'F7','F8','T3','T4',
        'T5','T6','Fz','Cz',
        'Pz','Movement'
    ]
    return channels[index]

def getRefAyden(index):
    localPath = '/Users/Ayden/Documents/BCI/ML_Training/'

    localFiles = [
        '5F-SubjectB-160309-5St-SGLHand-HFREQ.csv',
        '5F-SubjectB-160311-5St-SGLHand-HFREQ.csv',
        '5F-SubjectC-160429-5St-SGLHand-HFREQ.csv',
        '5F-SubjectF-160210-5St-SGLHand-HFREQ.csv',
        '5F-SubjectG-160413-5St-SGLHand-HFREQ.csv',
        '5F-SubjectH-160804-5St-SGLHand-HFREQ.csv',
        '5F-SubjectI-160719-5St-SGLHand-HFREQ.csv',
        'CLA-SubjectJ-170504-3St-LRHand-Inter.csv',
        'CLA-SubjectJ-170508-3St-LRHand-Inter.csv',
        'CLA-SubjectJ-170510-3St-LRHand-Inter.csv',
        '5F-SubjectI-160723-5St-SGLHand-HFREQ.csv',
        '5F-SubjectE-160321-5St-SGLHand-HFREQ.csv'
    ]
    return (localPath + localFiles[index])

def getRefJosh(index):
    localPath = ''

    localFiles = [
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        '',
    ]

    return (localPath + localFiles[index])