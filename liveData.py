import argparse
import time
import numpy as np

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations


def main():
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

    while True:
        
        # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
        data = board.get_current_board_data(1)  # get all data and remove it from internal buffer
        print(data[eeg_chan])

if __name__ == "__main__":
    main() 