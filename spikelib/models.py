"""Set of lineal and no lineal neuron model.

For example Spike Triggered Average (STA) is a algorithm to compute a lineal
filter to estimate receptive field of a cell.
"""
from functools import partial
from multiprocessing import cpu_count
from multiprocessing import Pool
from multiprocessing import freeze_support

import numpy as np

from spikelib.io import load_stim_multi


# A global dictionary storing the variables passed from the initializer.
GLOBAL_STIM = {}


def ste(stim, stim_time, spikes, nsamples_before=30, nsamples_after=0):
    """
    Get all windows of stimulis triggered by a spike.

    This function create a iterator to get a set of stimulus for a
    spike.

    Parameters
    ----------
    time_stim : ndarray
        The time array corresponding to the start of each frame in
        the stimulus.

    stimulus : ndarray
        A spatiotemporal or temporal stimulus array, where time is the
        first dimension.

    spikes : ndarray
        A list or ndarray of spike times.

    nsamples_before : int
        Number of samples to include in the STE before the spike.

    nsamples_after : int defaults: 0
        Number of samples to include in the STE after the spike.

    Returns
    -------
    ste : generator
        A generator that yields samples from the spike-triggered ensemble.

    Notes
    -----
    The spike-triggered ensemble (STE) is the set of all stimuli immediately
    surrounding a spike. If the full stimulus distribution is p(s), the STE
    is p(s | spike).

    """
    msg = 'time_stim.shape[0] must be equal than stim.shape[0]'
    assert stim.shape[0] == stim_time.size[0], msg

    bins_stim = np.append(stim_time, [stim_time[-1]*2 - stim_time[-2]])
    nbefore, nafter = nsamples_before, nsamples_after
    len_stim = stim.shape[0]
    # Number of spikes in each frame of the stimulus
    (nspks_in_frames, _) = np.histogram(spikes, bins=bins_stim)

    valid_frames = np.where(nspks_in_frames > 0)[0]
    filter_valid_fame = (valid_frames >= nbefore) & \
                        (valid_frames < len_stim - nafter)
    valid_frames = valid_frames[filter_valid_fame]
    spike_in_frames = nspks_in_frames[valid_frames]

    # Valid frames consider itself as reference
    for kfr, nspks in zip(valid_frames, spike_in_frames):
        yield nspks*stim[kfr+1-nbefore:kfr+1+nafter, :, :].astype('float64')


def sta(stim, stim_time, spikes, nsamples_before=30, nsamples_after=0):
    """
    Compute a spike-triggered average.

    Parameters
    ----------
    stim : ndarray
        A spatiotemporal or temporal stimulus array, where time is the
        first dimension.

    stim_time : ndarray
        The time array corresponding to the start of each frame in
        the stimulus.

    spikes : ndarray
        A list or ndarray of spike times

    nsamples_before : int
        Number of samples to include in the STA before the spike

    nsamples_after : int
        Number of samples to include in the STA after the spike (default: 0)

    Returns
    -------
    sta : ndarray
        The spatiotemporal spike-triggered average.

    References
    ----------
    A simple white noise analysis of neuronal light responses.
    E J Chichilnisky

    """
    nframe_stim, ysize, xsize = stim.shape
    sta_array = np.zeros((nsamples_before+nsamples_after, ysize, xsize))

    ste_it = ste(stim, stim_time, spikes, nsamples_before, nsamples_after)
    for kwindow_stim in ste_it:
        sta_array += kwindow_stim
    if sta_array.any():
        sta_array /= float(spikes.size)
    return sta_array


def multi_sta(spiketimes, stim_time, nsamples_before=30, nsamples_after=0):
    """Compute the Spike Triggered Average for a cell.

    Parameter
    ---------
    spiketimes: tuple
        tuple(name, spiketimes) to compute STA
    stim_time: array
        array with start time of each frame of stim
    nsamples_before : int
        Number of samples to include in the STA before the spike
    nsamples_after : int
        Number of samples to include in the STA after the spike (default: 0)

    Returns
    ---------
    unit_name:  str
        name of the unit
    sta_array: ndarray
        STA array

    """
    unit_name, spk_time = spiketimes

    stim_matrix = np.frombuffer(
        GLOBAL_STIM['stim']).reshape(GLOBAL_STIM['stim_shape'])
    nframe_stim, ysize, xsize = stim_matrix.shape

    bins_stim = np.append(stim_time, [stim_time[-1]*2 - stim_time[-2]])
    nspikes_in_frame, _ = np.histogram(spk_time, bins=bins_stim)

    valid_frames = np.where(nspikes_in_frame > 0)[0]
    filter_valid_fame = (valid_frames >= nsamples_before) & \
                        (valid_frames < nframe_stim - nsamples_after)
    valid_frames = valid_frames[filter_valid_fame]
    spike_in_frames = nspikes_in_frame[valid_frames]

    nframes_sta = nsamples_before+nsamples_after
    sta_array = np.zeros((nframes_sta, ysize, xsize), dtype=np.float64)
    # Valid frames consider itself as reference
    for kframe, nspikes in zip(valid_frames, spike_in_frames):
        start_frame = kframe-nsamples_before+1
        end_frame = kframe+nsamples_after+1
        sta_array += nspikes*stim_matrix[start_frame:end_frame, :, :]
    if sta_array.any():
        sta_array /= spike_in_frames.sum()
    return (unit_name, sta_array)


def init_multi_sta(stim, stim_shape):
    """Set stim array to a global variable."""
    GLOBAL_STIM['stim'] = stim
    GLOBAL_STIM['stim_shape'] = stim_shape


def run_multi_sta(stim_path, stim_time, spiketimes, nsamples_before=30,
                  nsamples_after=0, normed_stim=True, channel_stim='g'):
    """
    Run sta in multiprocessing.

    Parameter
    ---------
    stim_path: str
        file of the stim
    stim_time: array
        times of start and end of stim
    spiketimes: dict
        spiketimes to compute sta

    Returns
    -------
    stats: list of tuple
        return a list of tuple with unit_name and sta_array

    """
    freeze_support()
    stim, stim_shape = load_stim_multi(stim_path, normed=normed_stim,
                                       channel=channel_stim,
                                       dataset='checkerboard',
                                       )
    print(stim_shape)
    wrap_sta = partial(multi_sta,
                       stim_time=stim_time,
                       nsamples_before=nsamples_before,
                       nsamples_after=nsamples_after,
                       )
    pool = Pool(processes=cpu_count(), initializer=init_multi_sta,
                initargs=(stim, stim_shape),
                )
    result = pool.map(wrap_sta, spiketimes)
    return result
