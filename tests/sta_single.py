import sys
import os
sys.path.append(os.path.realpath('..'))

import h5py
import numpy as np
import matplotlib.pyplot as plt

from mealib.analysis import sta

source_folder = '/home/cesar/exp/MEA-analysis/data/'
stim_file = source_folder+'stim/checkerboard/stim_mini_MR-0227_.hdf5'
sorting_file = source_folder+'sorting/2018-01-25/2018-01-25.result.hdf5'
sync_file = source_folder+'sync/MR-0227/event_list/028.txt'

sync_times = np.loadtxt(sync_file).T
start_frame = sync_times[0]
end_frame = sync_times[1]
start_sync = start_frame[0]
end_sync = end_frame[-1]
bins_stim = np.concatenate((start_frame, end_frame[-1:]))
len_sync = len(sync_times)

stim_matrix = sta.load_stim_hdf5(stim_file)
unit_name = 'temp_71'
with h5py.File(sorting_file) as sorting:
    timestamp = sorting['/spiketimes/'+unit_name][...]
    ts_checkerboard = sta.get_times_for_sta(timestamp, start_sync, end_sync)

sta_array = sta.single_sta(stim_matrix, ts_checkerboard, bins_stim,
                           pre_frame=30, post_frame=0)
fig, ax = sta.plot_sta(sta_array, unit_name)
plt.show()
