import csv
import time

from .Other.determiningFeatures import getRowFromBucket
from .Other.utilFunctions import fillBucket

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

    def setTime(self):
        colLength = len(self.data[0])
        self.time = colLength / self.frequency

    def setMl_XY(self):

        x = []
        y = []

        tic = time.perf_counter()
        print("Populating feature matrix & Y vector...")
        dataRowEntries = 0
        numOfZeroes = 0

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
        print(f"{dataRowEntries} Data entries have been filled in {toc - tic:0.4}s!")