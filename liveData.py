import time
import numpy as np

from scipy.signal import butter, lfilter
from scipy import signal
from scipy.integrate import simps

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

import matplotlib.pyplot as plt

rawData=[ [],[],[],[],[],[],[],[] ]
filteredData=[ [],[],[],[],[],[],[],[] ]
def bandpass_filter(data, timeWindow, lowcut, highcut, fs, order=5):
    nyq =  fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    for i in range(0,len(rawData)):
        rawData[i].append(data[i][0])
        if len(rawData[i]) > timeWindow*fs:
            startIndex = len(rawData[i]) - (timeWindow*fs)
            endIndex = len(rawData[i]) - 1
            filteredData[i] = lfilter(b,a ,rawData[i][startIndex:endIndex])
        else:
            startIndex = 0
            endIndex = len(rawData[i]) - 1
            filteredData[i] = lfilter(b, a, rawData[i][startIndex:endIndex])

    return filteredData

alphaPSD = [ [],[],[],[],[],[],[],[] ]
def alphaPower(filteredData, fs, timeWindow):
    
    low, high = 8, 13
    window = timeWindow 
    for i in range(0,len(filteredData)):
        f, psd = signal.welch(filteredData[i],fs, nperseg=window)
        alphaBand = np.logical_and(f >= low, f <= high)

        fRes = f[1] - f[0] 
    
        alpha_power = simps(psd[alphaBand], dx=fRes)
        
        alphaPSD[i].append(alpha_power)
    
    return alphaPSD

betaPSD = [ [],[],[],[],[],[],[],[] ]
def betaPower(filteredData, fs, timeWindow):
    low, high = 8, 13
    window = timeWindow 
    for i in range(0,len(filteredData)):
        f, psd = signal.welch(filteredData[i],fs, nperseg=window)
        betaBand = np.logical_and(f >= low, f <= high)

        fRes = f[1] - f[0] 
    
        beta_power = simps(psd[betaBand], dx=fRes)
        
        betaPSD[i].append(beta_power)
    
    return betaPSD

def extractFeatures(filteredData):
    featureArray = []
 
    return featureArray

def main():
    # Bandpass Filter Variables
    fs = 250
    lowcut = 8.0
    highcut = 30.0
    order = 5
    timeWindow = 25 #Window of time where samples of data are used to calculate features

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
    timeout = time.time() + 4
    x = []
    i = 0
    while True: #len(rawData[0]) < 1000:
        i += 1
        x.append(i)
        # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
        data = board.get_current_board_data(1)[eeg_chan]*SCALE_FACTOR_EEG  # get all data and remove it from internal buffer              
        #Filters data based on the last 1000 samples
        filteredData = bandpass_filter(data, timeWindow, lowcut,highcut,fs,order)
        if len(filteredData[0]) > 25:
            alphaPSD = alphaPower(filteredData, fs, timeWindow)
            betaPSD = betaPower(filteredData, fs, timeWindow)
        #extractFeatures(filteredData)
        if time.time() > timeout:
            break 
          

    toc = time.perf_counter()
    print(f'Filtered {len(alphaPSD[0])} samples in {toc - tic:0.4}s!')
    print(alphaPSD[0]) 
    plt.plot(x[0:len(filteredData[0])],filteredData[0])
    plt.show()
        #Parse data so it is organized in column vectors
        #Filter each column 
        #apply feature extraction to each column 

if __name__ == "__main__":
    t = np.arange(0,10,0.1)
    y = np.sin(t*20)

    f, psd = signal.welch(y , 10)
    alphaBand = np.logical_and(f >= 8, f <= 13)

    fRes = f[1] - f[0] 
    print(f)
    print(psd[alphaBand])
   
    #alpha_power = simps(psd[alphaBand], dx=fRes)
    
    #plt.plot(x,alpha_power)
    