import time
import numpy as np
from scipy.signal import butter, lfilter

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

rawData=[ [],[],[],[],[],[],[],[] ]
filteredData=[ [],[],[],[],[],[],[],[] ]
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    for i in range(0,len(rawData)):
        rawData[i].append(data[i][0])
        if len(rawData[i]) > 10000:
            startIndex = len(rawData[i]) - 10000
            endIndex = len(rawData[i]) - 1
            filteredData[i] = lfilter(rawData[i][startIndex:endIndex])
        else:
            startIndex = 0
            endIndex = len(rawData[i]) - 1
            filteredData[i] = lfilter(b, a, rawData[i][startIndex:endIndex])

    return filteredData

def extractFeatures(filteredData):
    featureArray = []
 
    return featureArray

def main():
    # Bandpass Filter Variables
    fs = 250
    lowcut = 8.0
    highcut = 30.0
    order = 5

    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = 'COM7'

    board = BoardShim(0, params)
    print(board.get_device_name(0))
    eeg_chan = BoardShim.get_eeg_channels(0)

    board.prepare_session()

    # board.start_stream () # use this for default options
    board.start_stream(9600)
    time.sleep(1)

    SCALE_FACTOR_EEG = (4500000)/24/(2**23-1) #uV/count


    tic = time.perf_counter()
    i=0
    while i < 100000:
        i+=1
        # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
        data = board.get_current_board_data(1)[eeg_chan]  # get all data and remove it from internal buffer              
        
        #Filters data based on the last 1000 samples
        filteredData = bandpass_filter(data,lowcut,highcut,fs,order)
        #extractFeatures(filteredData)
   
    toc = time.perf_counter()
    print(f'Filtered 100000 samples in {toc - tic:0.4}s!')
        #Parse data so it is organized in column vectors
        #Filter each column 
        #apply feature extraction to each column 

    print(len(filteredData[7]))

if __name__ == "__main__":
    main() 