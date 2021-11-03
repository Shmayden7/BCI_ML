from .Classes.TrainingData import TrainingData
from .Classes.Other.utilFunctions import getFileRef


def createDataSet(userID, numOfInstances, nullPercentage):
    instanceArray = []
    print(f"\nCreating a data set from {numOfInstances} CSV files...")
    if numOfInstances <= 9:
        for x in range(numOfInstances):
            instanceArray.append(TrainingData(getFileRef(x,userID), nullPercentage))

    return instanceArray
# Needs to be tested
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
       