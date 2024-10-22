# imports 
##################################
import csv
import time

from scipy.sparse import data

from .Other.determiningFeatures import getRowFromBucket
from .Other.preProcessing import bandpassFilter
from .Other.featureFunctions import bandPower, waveletTransformProps, SampEnt, hMob, hCom, autoRegCoeff
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

    def __init__(self, filePath, featureID, nullPercentage = 0.3):
        self.filePath = filePath
        self.nullPercentage = nullPercentage
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
        numOfOnes = 0
        numOfTwos = 0
        numOfFours = 0
        numOfSixes = 0
        timeFrame = 10
        window = timeFrame * self.frequency

        tic1 = time.perf_counter()
        print("Populating feature matrix & Y vector...")

        # print(f'Length of C1 Before: {len(self.data[0])}')
        # print(f'Length of Markers Before: {len(self.data[len(self.data) - 1])}')
        # print(f'First C1 Val Before: {self.data[0][0]}')

        # Appending Power to the end of Data  
        totalColumns = len(self.data) # we need this since self.data is being updated
        for col in range(totalColumns - 1):
            tic2 = time.perf_counter()
            print(f'Current Col: {col}')
            alphaCol, betaCol, cA_max, cA_min, cA_mean, cA_median, cA_stDev, cD_max, cD_min, cD_mean, cD_median, cD_stDev, sampEnt, hMobility, hComplexity, autoReg = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
            for row in range(window, int(len(self.data[col]) / 4)):
            # for row in range(window, window + 100): # TESTING
                if (row % 100000) == 0: 
                    print(f'Current Row: {row}')
                if row < window:
                    start = 0
                    end = row
                else:
                    start = row - window
                    end = row

                # Adding features to the ml_X attributes
                tic3 = time.perf_counter()

                #(self.data[col][start:end])
                if self.featureID == 1:
                    alphaCol.append(bandPower(self.data[col][start:end], 'alpha', self.frequency, timeFrame))
                    betaCol.append(bandPower(self.data[col][start:end], 'beta', self.frequency, timeFrame))
                
                    props = waveletTransformProps(self.data[col][start:end])

                    cA_max.append(props['cA']['max'])
                    cA_min.append(props['cA']['min'])
                    cA_mean.append(props['cA']['mean'])
                    cA_median.append(props['cA']['median'])
                    cA_stDev.append(props['cA']['stDev'])

                    cD_max.append(props['cD']['max'])
                    cD_min.append(props['cD']['min'])
                    cD_mean.append(props['cD']['mean'])
                    cD_median.append(props['cD']['median'])
                    cD_stDev.append(props['cD']['stDev'])

                elif self.featureID == 2:
                    hMobility.append(hMob(self.data[col][start:end]))
                    hComplexity.append(hCom(self.data[col][start:end]))

                    autoReg.append(autoRegCoeff(self.data[col][start:end]))

                elif self.featureID == 5:
                    betaCol.append(bandPower(self.data[col][start:end], 'beta', self.frequency, timeFrame))

                    hMobility.append(hMob(self.data[col][start:end]))

                    props = waveletTransformProps(self.data[col][start:end])

                    cA_max.append(props['cA']['max'])
                    cA_min.append(props['cA']['min'])
                    cA_stDev.append(props['cA']['stDev'])

                    cD_max.append(props['cD']['max'])
                    cD_min.append(props['cD']['min'])
                    cD_stDev.append(props['cD']['stDev'])



                #sampEnt.append(sampEnt(self.data[col][start:end]))

                toc3 = time.perf_counter()

                print(f"Calculated features in {toc3 - tic3}s")

            # Appending columns to data attribute
            if self.featureID == 1:
                self.data.append(alphaCol)
                self.data.append(betaCol)

                self.data.append(cA_max)
                self.data.append(cA_min)
                self.data.append(cA_mean)
                self.data.append(cA_median)
                self.data.append(cA_stDev)
                
                self.data.append(cD_max)
                self.data.append(cD_min)
                self.data.append(cD_mean)
                self.data.append(cD_median)
                self.data.append(cD_stDev)
            
            elif self.featureID == 2:
                self.data.append(hMobility)
                self.data.append(hComplexity)
                self.data.append(autoReg)

            elif self.featureID == 5:
                self.data.append(betaCol)

                self.data.append(hMobility)

                self.data.append(cA_max)
                self.data.append(cA_min)
                self.data.append(cA_stDev)

                self.data.append(cD_max)
                self.data.append(cD_min)
                self.data.append(cD_stDev)

          

            # Removing the top rows of eeg data for each channel, 'window' long
            self.data[col] = self.data[col][window: int(len(self.data[col]) / 4)]

            toc2 = time.perf_counter()
            print(f'Column {col} filled in {toc2 - tic2:0.4}s')

            print(f'Number of Entries in column {col}: {len(self.data[col])}')
            print(f'Total Number of Columns: {len(self.data)}')

        # Move Y col from middle to end of self.data
        tempCol = self.data[8]
        self.data.pop(8)
        self.data.append(tempCol)

        # Removing the top rows of 'window' length from marker col
        self.data[len(self.data) - 1] = self.data[len(self.data) - 1][window:int(len(self.data[len(self.data) - 1]) / 4)]

        ####################################
        # Feature Selection
        ####################################

        print(f'Length of C1 After: {len(self.data[0])}')
        print(f'Length of Markers After: {len(self.data[len(self.data) - 1])}')
        print(f'Length of Last Feature Channel After: {len(self.data[len(self.data) - 2])}')

        # Removing Null Percentage
        for row in range(len(self.data[0])):
            currentRow = []

            # Getting the marker
            currentMarkerValue = self.data[len(self.data)-1][row]

            if currentMarkerValue == 0 or currentMarkerValue == 91 or currentMarkerValue == 92 or currentMarkerValue == 99 or currentMarkerValue == 3 or currentMarkerValue == 5:
                returnMarkerValue = 0
                numOfZeroes += 1
            elif currentMarkerValue == 1: 
                numOfOnes += 1
                returnMarkerValue = currentMarkerValue
            elif currentMarkerValue == 2: 
                numOfTwos += 1
                returnMarkerValue = currentMarkerValue
            elif currentMarkerValue == 4: 
                numOfFours += 1
                returnMarkerValue = currentMarkerValue
            elif currentMarkerValue == 6: 
                numOfSixes += 1
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

        toc1 = time.perf_counter()
        self.ml_X= x
        self.ml_y  = y
        print(f'Num of Rows in ml_X: {len(self.ml_X)}')
        print(f'Num of Cols in ml_x: {len(self.ml_X[0])}')
        print(f'Num of Rows in ml_y: {len(self.ml_y)}\n')

        print(f'Num of Zeroes: {numOfZeroes}')
        print(f'Num of Ones: {numOfOnes}')
        print(f'Num of Twos: {numOfTwos}')
        print(f'Num of Fours: {numOfFours}')
        print(f'Num of Sixes: {numOfSixes}\n')

        print(f"{dataRowEntries} Data entries have been filled in {toc1 - tic1:0.4}s!")