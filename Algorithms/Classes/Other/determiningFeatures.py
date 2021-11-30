from .featureFunctions import mav, aveOfCol, maxDiff

def divideBucket(currentBucket, divisionID):

    if divisionID == 1:
        eegChannels = currentBucket[range(0,8),:]
        alphaPower = currentBucket[range(8,17),:]  
        betaPower = currentBucket[range(17, 24),:] 
        bothPowers = currentBucket[range(8,24),:]

        divisions = {
            'eegChannels': eegChannels,
            'alphaPower': alphaPower,
            'betaPower': betaPower,
            'bothPowers': bothPowers
        }

    if divisionID == 2:
        onlyDivision = currentBucket

        divisions = {
            'onlyDivision': onlyDivision
        }
    
    return divisions


def determineFeaturesFromDivision(divisions, divisionID, featureID):

    if (divisionID == 1 and featureID == 1):
        aveVol = aveOfCol(divisions['eegChannels'])
        mavOfVol = mav(divisions['eegChannels'])
        aveAlphaPower = aveOfCol(divisions['alphaPower'])
        aveBetaPower = aveOfCol(divisions['betaPower'])
        diffOfPower = maxDiff(divisions['bothPowers'])

        rowOfFeatures = aveVol + mavOfVol + aveAlphaPower + aveBetaPower + diffOfPower

    if (divisionID == 2 and featureID == 1):
        rowOfFeatures = divisions['onlyDivision']
        
    return rowOfFeatures


def getRowFromBucket(currentBucket, divisionID, featureID):
    divisions = divideBucket(currentBucket, divisionID)
    rowOfFeatures = determineFeaturesFromDivision(divisions, divisionID, featureID)
    return rowOfFeatures

