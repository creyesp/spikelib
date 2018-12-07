"""Tools to manipulate stumulus."""

import numpy as np
import h5py

from spikelib.io import load_matstim
from spikelib.utils import check_groups


def correct_checkerboard(stimpath, syncpath, repeatedpath, outputpath,
                         matvar='stim', output_group='/',
                         output_dataset='checkerboard',
                         ):
    """Create a new stimulus with all repeated frames.

    Take a checkerboar stimulus and add all repeated frame found in a
    experiment and create a new stim file.

    Parameters
    ----------
    stimpath :
        path to original stim file (.mat).
    syncpath : str
        path to the syncronization file with start and end time for
        checkerboard (txt format).
    repeatedpath :
        path to file with all repeated times of experiment
        (txt format).
    outputpath :
        path to save stim to hdf5 file.

    Note
    ----------
    scipy.io.loadmat read matfile in mantain the matlab axis order
    to access array, ej. shape (y,x,channel,frame) = (35,35,3,72000)
    and python should be (frame,y,x,channel) = (72000,35,35,3), for
    this reason the output file keep python format.
    https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays/
    http://scikit-image.org/docs/dev/user_guide/numpy_images.html

    """
    stim = load_matstim(stimpath, matvar=matvar)

    print('Shape for checkerboar file: {}'.format(stim.shape))
    sync_frame = np.loadtxt(syncpath)
    repetared_frame = np.loadtxt(repeatedpath)

    repeated = np.where(np.isin(sync_frame[:, 0], repetared_frame))[0]
    n_repeated = len(repeated)

    # Find repeated position an number of repetitions
    if n_repeated > 1:
        rep_pointer = 0
        counter = {repeated[rep_pointer]: 1}
        for krep in range(1, len(repeated)):
            if (repeated[krep] - repeated[krep-1]) > 1:
                counter[repeated[krep]] = 1
                rep_pointer = krep
            else:
                counter[repeated[rep_pointer]] += 1

    elif n_repeated == 1:
        counter = {repeated[0]: 1}
    else:
        counter = {}

    # Correct of repeated to get the original stim repeated
    sort_keys = [k for k in counter.keys()]
    sort_keys.sort()
    print('Repeated frame {}'.format(sort_keys))

    if len(counter) > 1:
        corrected_repeated = {sort_keys[0]: counter[sort_keys[0]]}
        corrected_sum = counter[sort_keys[0]]
        for krep in range(1, len(sort_keys)):
            kkey = sort_keys[krep]
            corrected_repeated[kkey-corrected_sum] = counter[kkey]
            corrected_sum += counter[kkey]
    elif len(counter) == 1:
        corrected_repeated = counter
        corrected_sum = counter[sort_keys[0]]
    else:
        corrected_repeated = counter
        corrected_sum = 0

    stim_shape = list(stim.shape)
    new_stim_shape = list(stim_shape)
    new_stim_shape[0] += corrected_sum

    new_stim = np.empty(tuple(new_stim_shape), dtype=np.uint8)

    range_stim = [k for k in corrected_repeated]
    range_stim.sort()
    range_stim = np.array([[0] + range_stim, range_stim + [stim_shape[0]]])
    range_stim = range_stim.transpose()

    if 0 not in corrected_repeated:
        corrected_repeated[0] = 0

    delay = 0
    for kstart, kend in range_stim:
        delay_rep = corrected_repeated[kstart]
        new_start, new_end = (kstart+delay, kstart+delay+delay_rep)
        new_stim[new_start:new_end, ...] = stim[kstart:kstart+1, ...]
        delay += delay_rep
        new_stim[kstart+delay:kend+delay, ...] = stim[kstart:kend, ...]
    del stim, sync_frame, repetared_frame

    with h5py.File(outputpath, 'a') as f:
        if not output_group.endswith('/'):
            output_group = output_group + '/'
        check_group(f, [output_group])

        if output_dataset in f[output_group]:
            f[output_group+output_dataset][...] = new_stim
        else:
            f.create_dataset(output_group+output_dataset,
                             data=new_stim,
                             dtype=np.uint8,
                             chunks=tuple([1]+new_stim_shape[1:]),
                             compression='gzip',
                             shuffle=False,
                             )
        f[output_group].attrs['nrepeated'] = corrected_sum
        repeated_str = ','.join(map(lambda x: str(x), sort_keys))
        f[output_group].attrs['repeated'] = repeated_str
        f[output_group].attrs['dim'] = u'(nframe,ysize,xsize,nchannel)'
