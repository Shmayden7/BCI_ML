import csv
import time
import numpy as np
import pywt
from scipy.sparse import data
from scipy.fft import fft

from .Other.determiningFeatures import getRowFromBucket
from .Other.utilFunctions import fillBucket

class TrainingData:

    frequency = 200
    time = 0
    filePath = ''
    numOfBuckets = 0
    nullPercentage = 0
    data = []
    # fft = []
    # wt = []
    ml_X = []
    ml_y = []
    divisionID = 0
    featureID = 0 

    def __init__(self, filePath, divisionID, featureID, nullPercentage = 0.3, numOfBuckets = 0):
        self.filePath = filePath
        self.numOfBuckets = numOfBuckets
        self.nullPercentage = nullPercentage
        self.divisionID = divisionID
        self.featureID = featureID

        # Updating frequency for high frequency files
        lastLetters = filePath[-9:-4]
        if (lastLetters == 'HFREQ'):
            self.frequency = 1000

        # fill the data attribute from the .csv file
        self.fillData()

        # determining transform properties
        # self.calculateTransforms()

        # set the total time for the file
        self.setTime()

        # setting the numOfBuckets optional param
        self.setNumOfBuckets()

        # set ML X matrix and y vector
        self.setMl_XY()


    def fillData(self):

        dataHolder = []
        print(f"\nPopulating data from: {self.filePath[-12:-4]}...")

        tic = time.perf_counter()
        # Importing csv data from local file
        with open(f'{self.filePath}', 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            
            # skips the first 2 rows of the data files (name + sample rate)
            next(csv_reader)
            next(csv_reader)

            for rowIndex, row in enumerate(csv_reader): # Row by Row of matrix
                colIndex = 0
                for colIndex in range(len(row)): # indices of each row 

                    if rowIndex == 0:
                        dataHolder.append([])

                    string = row[colIndex] # Voltage as a string
                    double = float(string) # convert string -> double 

                    if (double == 0.0 or double == -0.0): # making the 0's look nice
                        double = 0

                    dataHolder[colIndex].append(double)
                    colIndex += 1

        toc = time.perf_counter()
        print(f'Data has been populated in {toc - tic:0.4}s!')
        self.data = dataHolder

    # def calculateTransforms(self):
    #     fftArray = []
        
    #     for col in range(len(self.data) - 1):
    #         fftArray.append(np.abs(fft(self.data[col])))
    #         #cA , cD = pywt.dwt(self.data[col])
    #         #wt.append(cA)
    #         #wt.append(cD)
    #     self.fft = fftArray
    #     # self.wt = wt

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
        for bucketNum in range(1):

            # Bucketing Data
            currentBucket = np.array(fillBucket(bucketNum)) 

            # Determining Markers
            markers = currentBucket[len(currentBucket) - 1,:] 

            if len(set(markers)) == 1:
                # Assigning value in y vector
                currentMarkerValue = list(set(markers))[0]
                returnMarkerValue = 0

                # Defining static and movement states (marker value)
                if currentMarkerValue == 0 or currentMarkerValue == 91 or currentMarkerValue == 92 or currentMarkerValue == 99 or currentMarkerValue == 3 or currentMarkerValue == 5:
                    returnMarkerValue = 0
                    numOfZeroes += 1
                else: 
                    returnMarkerValue = currentMarkerValue

                #Check if null percentage has been reached
                if numOfZeroes / self.numOfBuckets >= self.nullPercentage and returnMarkerValue == 0:
                    continue #skips current iteration

                rowOfFeatures = getRowFromBucket(currentBucket, self.divisionID, self.featureID)

                x.append(rowOfFeatures)
                y.append(returnMarkerValue)
                filledBuckets += 1 

        toc = time.perf_counter()
        print(f"{filledBuckets} buckets have been filled in {toc - tic:0.4}s!")
        self.ml_X = x
        self.ml_y = y
