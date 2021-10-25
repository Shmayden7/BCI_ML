import csv
import time
import numpy as np

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

def sampleRateDelay(sampleRate, value):   #Delays returning the next data point from the csv file based on the sample rate of the data  
    time1 = time.time()
    while True:
        if time.time() > (time1 + 1/sampleRate):  # check to see if a tenth of a second has passed
            break

    return value

def MAV(data):
    mean = np.mean(data)
    print(mean)
    totalDistance = 0
    for y in range(len(data)):
        totalDistance = totalDistance + abs(data[y]-mean)
    mav = totalDistance / len(data)
    return mav


