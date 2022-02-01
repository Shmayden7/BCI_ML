# imports 
##################################
# Packages
import time
from turtle import color
import matplotlib.pyplot as plt

#Live Data
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

#Relative Functions
from Algorithms.preProcessing import bandpassFilter
from Algorithms.Classes.Other.featureFunctions import bandPower, waveletTransformProps, SampEnt, hMob, hCom, autoRegCoeff
##################################

def cutoffData(filteredData, timeFrame):
    fs = 250

    if len(filteredData[0]) <= timeFrame * fs:
        return filteredData
    else:
        start = len(filteredData[0]) - timeFrame*fs
        end = len(filteredData[0]) -1
        filteredData[0] = filteredData[0][start:end]
        filteredData[1] = filteredData[1][start:end]
        filteredData[2] = filteredData[2][start:end]
        filteredData[3] = filteredData[3][start:end]
        filteredData[4] = filteredData[4][start:end]
        filteredData[5] = filteredData[5][start:end]
        filteredData[6] = filteredData[6][start:end]
        filteredData[7] = filteredData[7][start:end]

        return filteredData

def makeFeatures(filteredData, featureID, timeFrame):
    fs = 250

    totalColumns = len(filteredData) # we need this since self.data is being updated
    
    for col in range(totalColumns):

        tic = time.perf_counter()

        alphaCol, betaCol, cA_max, cA_min, cA_mean, cA_median, cA_stDev, cD_max, cD_min, cD_mean, cD_median, cD_stDev, sampEnt, hMobility, hComplexity, autoReg = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

        if featureID == 1:
            alphaCol.append(bandPower(filteredData[col], 'alpha', fs, timeFrame))
            betaCol.append(bandPower(filteredData[col], 'beta', fs, timeFrame))
        
            props = waveletTransformProps(filteredData[col])

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

        elif featureID == 2:
            hMobility.append(hMob(filteredData[col]))
            hComplexity.append(hCom(filteredData[col]))

            autoReg.append(autoRegCoeff(filteredData[col]))

        elif featureID == 5:
            betaCol.append(bandPower(filteredData[col], 'beta', fs, timeFrame))

            hMobility.append(hMob(filteredData[col]))

            props = waveletTransformProps(filteredData[col])

            cA_max.append(props['cA']['max'])
            cA_min.append(props['cA']['min'])
            cA_stDev.append(props['cA']['stDev'])

            cD_max.append(props['cD']['max'])
            cD_min.append(props['cD']['min'])
            cD_stDev.append(props['cD']['stDev'])


        toc = time.perf_counter()
        #print(f"Calculated features in {toc - tic}s")

        if featureID == 1:
            filteredData.append(alphaCol)
            filteredData.append(betaCol)

            filteredData.append(cA_max)
            filteredData.append(cA_min)
            filteredData.append(cA_mean)
            filteredData.append(cA_median)
            filteredData.append(cA_stDev)
            
            filteredData.append(cD_max)
            filteredData.append(cD_min)
            filteredData.append(cD_mean)
            filteredData.append(cD_median)
            filteredData.append(cD_stDev)

        elif featureID == 2:
            filteredData.append(hMobility)
            filteredData.append(hComplexity)
            filteredData.append(autoReg)

        elif featureID == 5:
            filteredData.append(betaCol)

            filteredData.append(hMobility)

            filteredData.append(cA_max)
            filteredData.append(cA_min)
            filteredData.append(cA_stDev)

            filteredData.append(cD_max)
            filteredData.append(cD_min)
            filteredData.append(cD_stDev)
            
                

    featureRow = []
    for col in filteredData:
        featureRow.append(col[len(col) - 1])

    del filteredData[8:]

    return featureRow
 
def main():
    # Bandpass Filter Variables
    sampleFreq = 250
    lowCut = 5.0
    highCut = 33.0
    order = 5
    timeFrame = 10 # Window of time where samples of data are used to calculate features
    featureID = 1
    # Connecting board to Python project
    ##################################
    BoardShim.enable_dev_board_logger() 

    params = BrainFlowInputParams()           # Initlizing the paramaters
    params.serial_port = 'COM7'               # The Port Where the USB is plugged in

    board = BoardShim(0, params)              # Tell you watboard your using
    eeg_chan = BoardShim.get_eeg_channels(0)  # Fetches 8 needed channels

    board.prepare_session()

    board.start_stream(9600)                  # Starts dataStream, IDK what number does
    time.sleep(1)                             # Wait 1 second for board to load fully                               

    SCALE_FACTOR_EEG = (4500000)/24/(2**23-1) #uV/count
    ##################################

    # Timing 5 seconds
    
    timeout = time.time() + 20

    x = []
    i = 0 # Used for plotting
    filteredData=[[],[],[],[],[],[],[],[]] 
    while time.time() < timeout: 
        i += 1
        x.append(i)   

        # Fetches all data and removes it from internal buffer
        liveData = board.get_current_board_data(1)[eeg_chan] #*SCALE_FACTOR_EEG    
        #liveData *= SCALE_FACTOR_EEG
        #Filters data based on timeFrame seconds
        
        filteredData = bandpassFilter(liveData, timeFrame, lowCut, highCut, sampleFreq, order)
        filteredData = cutoffData(filteredData, timeFrame)

       
        tic = time.perf_counter()

        if len(filteredData[0]) >= sampleFreq * 2:

            featureRow = makeFeatures(filteredData, featureID, 2)

            toc = time.perf_counter()

            print(f'Made features in {toc - tic:0.4}s!')
        
        # Create Feature Row
    print(len(featureRow))
    board.release_session()
        
    print(f'Length of filteredData: {len(filteredData)}')

    colors = ['red','blue', 'black','orange', 'sienna','green','magenta','cyan']
    fig, axs = plt.subplots(8)
    for i in range(len(filteredData)):
        print(f'Col {i} length: {len(filteredData[i])}')
        axs[i].plot(x[:len(filteredData[i])],filteredData[i], color = colors[i])

    plt.show()
        #Parse data so it is organized in column vectors
        #Filter each column 
        #apply feature extraction to each column 

if __name__ == "__main__":
    main()