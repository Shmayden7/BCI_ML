# imports 
##################################
# Packages
import time
import matplotlib.pyplot as plt

#Live Data
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

#Relative Functions
from .preProcessing import bandpassFilter, bandPower, extractFeatures
##################################

def main():
    # Bandpass Filter Variables
    sampleFreq = 250
    lowcut = 8.0
    highcut = 30.0
    order = 5
    timeFrame = 25 # Window of time where samples of data are used to calculate features

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
    tic = time.perf_counter()
    timeout = time.time() + 5

    x = []
    i = 0 # Used for plotting
    while True: #len(rawData[0]) < 1000: breaks after 5s
        i += 1
        x.append(i)   

        # Fetches all data and removes it from internal buffer
        liveData = board.get_current_board_data(1)[eeg_chan]*SCALE_FACTOR_EEG    

        #Filters data based on timeFrame seconds
        filteredData = bandpassFilter(liveData, timeFrame, lowcut, highcut, sampleFreq, order)
        
        if len(filteredData[0]) % 6 == 0:

        # Bucketing our Data


        # Create Feature Row
        # [raw_1, raw_1, bp_1, bp_2, ]
        
    
        if time.time() > timeout:
            break 
          

    toc = time.perf_counter()
    print(f'Filtered {len(filteredData[0])} samples in {toc - tic:0.4}s!')
    plt.plot(x[0:len(filteredData[0])],filteredData[0])
    plt.show()
        #Parse data so it is organized in column vectors
        #Filter each column 
        #apply feature extraction to each column 

if __name__ == "__main__":
    main()