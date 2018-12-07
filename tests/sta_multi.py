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

# Sync
sync_times = np.loadtxt(sync_file).T
start_frame = sync_times[0]
end_frame = sync_times[1]
start_sync = start_frame[0]
end_sync = end_frame[-1]
bins_stim = np.concatenate((start_frame, end_frame[-1:]))

# Spikes
fspiketimes = h5py.File(sorting_file)
list_name = [kname for kname in fspiketimes['spiketimes']]
spiketimes = []
for key in list_name[:10]:
    timestamp = fspiketimes['/spiketimes/'+key][...]
    ts_checkerboard = sta.get_times_for_sta(timestamp, start_sync, end_sync)
    spiketimes.append((key, ts_checkerboard))
fspiketimes.close()


result = sta.run_multi_sta(stim_file, bins_stim, spiketimes,
                           pre_frame=18, post_frame=0)
print('Results (pool):\n', len(result))

for (kname, ksta) in result:
    fig, ax = sta.plot_sta(ksta, kname)
plt.show()
