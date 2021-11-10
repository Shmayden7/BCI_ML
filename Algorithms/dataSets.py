import pickle
import random
from .Classes.Other.utilFunctions import sets

from .Classes.TrainingData import TrainingData
from .Classes.Other.utilFunctions import getCSVRef, getClassifierRef

def createDataSet(userID, divisionID, featureID, nullPercentage, numOfCSVs, startingPoint=0):
    instanceArray = []
    print(f"\nCreating a data set from {numOfCSVs} CSV files...")
    if numOfCSVs <= 10:
        for x in range(startingPoint, (numOfCSVs + startingPoint)):
            instanceArray.append(TrainingData(getCSVRef(x,userID,divisionID), divisionID, featureID, nullPercentage))

    return instanceArray

randomNumbers = [] 
def createRandomDataset(userID, divisionID, featureID, nullPercentage, numOfCSVs, startingPoint = 0):
    instanceArray = []
    print(f"\nCreating a random data set from {numOfCSVs} CSV files...")
    
    if numOfCSVs <= len(sets[divisionID]):
        x = 0
        while x < numOfCSVs:
            r = random.randint(0, len(sets[divisionID]) - 1)
            if r not in randomNumbers:
                x = x + 1
                randomNumbers.append(r)
                instanceArray.append(TrainingData(getCSVRef(r,userID,divisionID), divisionID, featureID, nullPercentage))

    return instanceArray

def mergeInstanceData(instanceArray):
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

    print("\nData has been merged!")
    print(f"Total Data Entries: {len(final_x)}")
    return final_x, final_y
 
def readAlgorithm(fileName, userID):
    filePath = getClassifierRef(userID)
    classifier = pickle.load(open((filePath + fileName), 'rb'))
    print(f'Classifier has loaded!')
    return classifier

def writeAlgorithm(fileName, classifier, userID):
    filePath = getClassifierRef(userID)
    with open((filePath + fileName), 'wb') as file:
        pickle.dump(classifier, file)

    print(f"\nWrote Classifier to: {filePath + fileName}")
