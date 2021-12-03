# imports 
##################################
import csv
import time

from scipy.sparse import data

from .Other.determiningFeatures import getRowFromBucket
from .Other.preProcessing import bandpassFilter
from .Other.featureFunctions import bandPower
##################################

class TrainingData:

    frequency = 200
    time = 0
    filePath = ''
    nullPercentage = 0
    data = []
    ml_X = []
    ml_y = []
    divisionID = 0
    featureID = 0 

    def __init__(self, filePath, divisionID, featureID, nullPercentage = 0.3):
        self.filePath = filePath
        self.nullPercentage = nullPercentage
        self.divisionID = divisionID
        self.featureID = featureID

        # Updating frequency for high frequency files
        lastLetters = filePath[-9:-4]
        if (lastLetters == 'HFREQ'):
            self.frequency = 1000

        # fill the data attribute from the .csv file
        self.fillData()

        # set the total time for the file
        self.setTime()

        # set ML X matrix and y vector
        self.setMl_XY()

    def fillData(self):

        dataHolder = []
        filteredData = []
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
                        dataHolder.append([]) # Appends empty col based on number of cols in csv

                    string = row[colIndex] # Voltage as a string
                    double = float(string) # convert string -> double 

                    if (double == 0.0 or double == -0.0): # making the 0's look nice
                        double = 0

                    dataHolder[colIndex].append(double)
                    colIndex += 1
        
        # Filtering each col of the data, not markers
        for col in range(len(dataHolder) - 1):
            filteredCol = bandpassFilter(dataHolder[col], sampleFreq=200, liveData=False)
            filteredData.append(filteredCol)
        
        # Adding markers to filteredData
        filteredData.append(dataHolder[len(dataHolder)-1])

        toc = time.perf_counter()
        print(f'Data has been populated in {toc - tic:0.4}s!')
        self.data = filteredData

    def setTime(self):
        colLength = len(self.data[0])
        self.time = colLength / self.frequency

    def setMl_XY(self):

        x = []
        y = []
        dataRowEntries = 0
        numOfZeroes = 0
        timeFrame = 30 

        tic = time.perf_counter()
        print("Populating feature matrix & Y vector...")

        # Appending Bandpower
        # [[8_eeg],[y],[a_power]

        # 1 1 1 1 1 1 1 1 y a
        # 1 1 1 1 1 1 1 1 y a
        # 1 1 1 1 1 1 1 1 y a
        # 1 1 1 1 1 1 1 1 y 
        # 1 1 1 1 1 1 1 1 y 
        # 1 1 1 1 1 1 1 1 y
        # 1 1 1 1 1 1 1 1 y

        # Appending Power to the end of Data  
        for col in range(len(self.data) - 1):
            bpCol = []
            for row in range(2,len(self.data[col])):
                if row < self.frequency*timeFrame:
                    start = 0
                    end = row
                else:
                    start = row - self.frequency*timeFrame
                    end = row
                bpCol.append(bandPower(self.data[col][start:end], 'alpha', self.frequency, timeFrame))
            self.data.append(bpCol)
        print(len(self.data))      

        # Move Y col from middle to end of self.data

        # Removing Null Percentage
        for row in range(len(self.data[0])):
            currentRow = []

            # Getting the marker
            currentMarkerValue = self.data[len(self.data)-1][row]

            if currentMarkerValue == 0 or currentMarkerValue == 91 or currentMarkerValue == 92 or currentMarkerValue == 99 or currentMarkerValue == 3 or currentMarkerValue == 5:
                    returnMarkerValue = 0
                    numOfZeroes += 1
            else: 
                returnMarkerValue = currentMarkerValue

            #Check if null percentage has been reached
            if numOfZeroes / len(self.data[0]) >= self.nullPercentage and returnMarkerValue == 0:
                continue #skips current iteration
            
            for col in range(len(self.data) - 1):
                currentRow.append(self.data[col][row])

            dataRowEntries += 1 

            # Appending Rows to X matrix and y vector
            y.append(returnMarkerValue)
            x.append(currentRow)

        toc = time.perf_counter()
        self.ml_X = x
        self.ml_y = y
        print(len(self.ml_x[0]))
        print(f"{dataRowEntries} Data entries have been filled in {toc - tic:0.4}s!")