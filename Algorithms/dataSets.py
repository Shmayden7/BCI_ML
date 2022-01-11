from os import read
import random
from sklearn.model_selection import train_test_split
from .Classes.Other.utilFunctions import sets, fileName

from .Classes.TrainingData import TrainingData
from .Classes.Other.utilFunctions import getCSVRef
from .Classes.Other.readWrite import readTrainingDataInstance

# Creating Datasets
##################################
def createDataSet(userID, divisionID, nullPercentage, numOfFiles, readPKL=True, startingPoint=0):
    instanceArray = []

    if numOfFiles <= len(fileName):

        if readPKL: # Reading from the pre-recorded pkl object instances
            print(f"\nCreating a data set from {numOfFiles} PKL files...")
            for x in range(startingPoint, (numOfFiles + startingPoint)):
                instanceArray.append(readTrainingDataInstance(x, userID, divisionID))
        else: # Reading from the raw csv files
            print(f"\nCreating a data set from {numOfFiles} CSV files...")
            for x in range(startingPoint, (numOfFiles + startingPoint)):
                instanceArray.append(TrainingData(getCSVRef(x,userID,divisionID), divisionID, nullPercentage))
    else:
        print('\nError! createDataSet failed, numOfFiles > available files')

    return instanceArray

randomNumbers = [] # placed outside the function so that if you run random for both training and testing
                   # you wont use the same csv's for both
def createRandomDataset(userID, divisionID, featureID, nullPercentage, numOfFiles, readPKL=True):
    instanceArray = []    
    if numOfFiles <= len(sets[divisionID]):
        x = 0
        if readPKL:
            print(f"\nCreating a random data set from {numOfFiles} PKL files...")
        else: 
            print(f"\nCreating a random data set from {numOfFiles} CSV files...")
            
        while x < numOfFiles:
            r = random.randint(0, len(sets[divisionID]) - 1)
            if r not in randomNumbers:
                x = x + 1
                randomNumbers.append(r)

                if readPKL: # Reading from the pre-recorded pkl object instances
                    instanceArray.append(readTrainingDataInstance(r, userID, divisionID))
                else: # Reading from the raw csv file
                    instanceArray.append(TrainingData(getCSVRef(r, userID, divisionID), divisionID, featureID, nullPercentage))

    return instanceArray
##################################
            
def mergeInstanceData(instanceArray, testSizePercentage, newData = False):
    final_x = []
    final_y = []

    # Filling final_x
    for instance in instanceArray:
        for row in instance.ml_X:
            final_x.append(row)

    # Filling final_y 
    for instance in instanceArray:
        for row in instance.ml_y:
            final_y.append(row)

    print("\nTrainingData instances have been merged!")
    print(f"Total Data Entries: {len(final_x)}")


    if newData == True:
        x_test = final_x
        y_test = final_y

        return x_test, y_test
    else:
        x_train, x_test, y_train, y_test = train_test_split(final_x, final_y, test_size=testSizePercentage, random_state=42)

        return x_train, x_test, y_train, y_test
