import csv

class TrainingData:

    frequency = 200
    data = []

    def __init__(self, filePath):
        self.filePath = filePath

        # Updating frequency for high frequency files
        lastLetters = filePath[-9:-4]
        if (lastLetters == 'HFREQ'):
            self.frequency = 1000

        # fill the data attribute from the .csv file
        self.fillData()

    def fillData(self):
        dataHolder = [
            [],[],[],[],[],[],[],[],
            [],[],[],[],[],[],[],[],
            [],[],[],[],[],[],[]
        ]

        # Importing csv data from local file
        with open(f'{self.filePath}', 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            print('Populating Matrix...')

            for line in csv_reader: # Row by Row of matrix
                row = 0
                for rowIndex in range(len(line)): # indices of each row 
                    string = line[rowIndex] # Voltage as a string
                    double = float(string) # convert string -> double 

                    if (double == 0.0 or double == -0.0): # making the 0's look nice
                        double = 0

                    #print(rowIndex)
                    dataHolder[rowIndex].append(double)
                    row = row+1
                    
            print(f'Data from {self.filePath} has been populated!')
