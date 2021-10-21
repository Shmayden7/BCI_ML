import csv
import time

# returns an array of int's from a .csv
def getBasicCsvData(filePath):
    dataVector = [] # return vector

    # Importing C3 amplitude from local .csv
    with open(f'{filePath}', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader) # skips the first index

        for line in csv_reader:
            string = line[0] # Voltage as a string
            split = string.split('.') # Array containing intiger and decimal
            number = int(split[0]) 
            dataVector.append(number)
        
    return dataVector

def sampleData(sampleRate, data):
    for i in range(len(data)):
        time1 = time.time()
        while True:
            if time.time() > (time1 + 1/sampleRate):  # check to see if a tenth of a second has passed
                break

        return data[i]



