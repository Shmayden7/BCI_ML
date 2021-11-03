import csv
import time
import numpy as np

from .Other.featureFunctions import mav, aveOfCol, maxDiff
from .Other.utilFunctions import fillBucket

class TrainingData:

    frequency = 200
    time = 0
    filePath = ''
    numOfBuckets = 0
    nullPercentage = 0
    data = []
    ml_X = []
    ml_y = []

    def __init__(self, filePath, nullPercentage = 0.3, numOfBuckets = 0):
        self.filePath = filePath
        self.numOfBuckets = numOfBuckets
        self.nullPercentage = nullPercentage

        # Updating frequency for high frequency files
        lastLetters = filePath[-9:-4]
        if (lastLetters == 'HFREQ'):
            self.frequency = 1000

        # fill the data attribute from the .csv file
        self.fillData()

        # set the total time for the file
        self.setTime()

        # setting the numOfBuckets optional param
        self.setNumOfBuckets()

        # set ML X matrix and y vector
        self.setMl_XY()


    def fillData(self):
        dataHolder = [
            [],[],[],[],[],[],[],[],[],[],
            [],[],[],[],[],[],[],[],[],[],
            [],[],[],[],[]
        ]

        tic = time.perf_counter()
        # Importing csv data from local file
        with open(f'{self.filePath}', 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            # skips the first 2 rows of the data files (name + sample rate)
            next(csv_reader)
            next(csv_reader)
            print('\nPopulating Matrix...')

            for col in csv_reader: # Row by Row of matrix
                row = 0
                for rowIndex in range(len(col)): # indices of each row 
                    string = col[rowIndex] # Voltage as a string
                    double = float(string) # convert string -> double 

                    if (double == 0.0 or double == -0.0): # making the 0's look nice
                        double = 0

                    #print(rowIndex)
                    dataHolder[rowIndex].append(double)
                    row = row+1
        toc = time.perf_counter()
        print(f'Data from {self.filePath[-12:]} has been populated in {toc - tic:0.4}s!')
        self.data = dataHolder

    def setTime(self):
        colLength = len(self.data[0])
        self.time = colLength / self.frequency

    def setNumOfBuckets(self):
        if self.numOfBuckets == 0:
            self.numOfBuckets = len(self.data[0]) // 6

    def setMl_XY(self):

        x = []
        y = []

        # Structure of x
        #  [f1_c1,f1_c2...,f2_c1,,,,,] b=1 (training example 1)
        #  [,,,,,,,] b=2
        #  [,,,,,,,] b=3
        #  [,,,,,,,] b=4
        tic = time.perf_counter()
        print("Populating feature matrix & Y vector...")
        filledBuckets = 0
        numOfZeroes = 0

        # Number of full buckets in instance
        for bucketNum in range(self.numOfBuckets):

            # Bucketing Data
            currentBucket = np.array(fillBucket(bucketNum, self.data)) 

            # Dividing info from bucket
            eegChannels = currentBucket[range(0,8),:]
            alphaPower = currentBucket[range(8,17),:]  
            betaPower = currentBucket[range(17, 24),:] 
            bothPowers = currentBucket[range(8,24),:]
            markers = currentBucket[24,:] 

            if len(set(markers)) == 1: # Checks if there are different marker values in bucket
                
                # Creating channels from info
                aveVol = aveOfCol(eegChannels)
                mavOfVol = mav(eegChannels)
                aveAlphaPower = aveOfCol(alphaPower)
                aveBetaPower = aveOfCol(betaPower)
                diffOfPower = maxDiff(bothPowers)

                # Assigning value in y vector
                currentValue = list(set(markers))[0]
                vectorValue = 0

                # Defining static and movement states
                if currentValue == 0 or currentValue == 91 or currentValue == 92 or currentValue == 99 or currentValue == 3 or currentValue == 5:
                    vectorValue = 0
                    numOfZeroes += 1
                else: 
                    vectorValue = currentValue
                
                #Check if null percentage has been reached
                if numOfZeroes / self.numOfBuckets >= self.nullPercentage and vectorValue == 0:
                    continue #skips current iteration
                    
                # Filling features for bucket, add row to matrix
                row = aveVol + mavOfVol + aveAlphaPower + aveBetaPower + diffOfPower
                x.append(row)

                y.append(vectorValue)
                filledBuckets += 1 

        toc = time.perf_counter()
        print(f"{filledBuckets} buckets have been filled in {toc - tic:0.4}s!")

        self.ml_X = x
        self.ml_y = y

