import csv
import time

class TrainingData:

    frequency = 200
    data = []
    time = 0
    filePath = ''

    def __init__(self, filePath):
        self.filePath = filePath

        # Updating frequency for high frequency files
        lastLetters = filePath[-9:-4]
        if (lastLetters == 'HFREQ'):
            self.frequency = 1000

        # fill the data attribute from the .csv file
        self.fillData()

        # set the total time for the file
        self.setTime()

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
            print('Populating Matrix...')

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
        print(f'Data from {self.filePath[-13:]} has been populated in {toc - tic:0.4}s!')
        self.data = dataHolder

    def setTime(self):
        colLength = len(self.data[0])
        self.time = colLength / self.frequency
