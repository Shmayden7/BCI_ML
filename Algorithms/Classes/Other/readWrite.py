import pickle
import time
from .utilFunctions import getClassifierRef, getPKLRef

# Reading / Writing Instances as .pkl
##################################
def readTrainingDataInstance(index, userID, divisionID):
    filePath = getPKLRef(index, userID, divisionID)
    tic = time.perf_counter()

    with open(filePath, 'rb') as pickleFile:
        instance = pickle.load(pickleFile)
        
    toc = time.perf_counter()
    print(f'Loaded: {instance.filePath[-12:-4]} in {toc - tic:0.4}s')
    return instance

def writeTrainingDataInstance(instanceArray):
    for instance in instanceArray:
        name = instance.filePath[-12:-4]

        with open(f'{name}.pkl', 'wb') as pickleFile:
            pickle.dump(instance, pickleFile)
            print(f'\nWrote {name} to a pickle file!')
##################################

# Reading / Writing Classifiers as .pkl
##################################
def readClassifier(fileName, userID):
    filePath = getClassifierRef(userID)
    classifier = pickle.load(open((filePath + fileName) + '.pkl', 'rb'))
    print(f'Classifier has loaded!')
    return classifier

def writeClassifier(fileName, classifier, userID):
    filePath = getClassifierRef(userID)
    with open((filePath + fileName), 'wb') as file:
        pickle.dump(classifier, file)

    print(f"\nWrote Classifier to: {filePath + fileName}")
##################################