"""Set of tools to manipulate input and output files."""
from multiprocessing import RawArray

import h5py
import numpy as np
from scipy.io import loadmat


def load_stim(filepath, normed=True, channel='g', dataset='checkerboard'):
    """
    Return checkerboard stim from hdf5 file.

    Read checkerboard stimulus from hdf5 file. It file was obtained
    from load_matstim() function and sava to a hdf5 file. The
    stimulus must has 4 dimension, nframe, yblock, xbock, nchannels.

    Parameters
    ----------
    filepath: str
        relative or full path of hdf5 file.
    normed: bool default: True
        Normalize stimulus between -1 and 1.
    channel: {'r', 'g', 'b'}
        stimulus has 3 channels, 'r', 'g' and 'b'. Set one of these
        to retrive stim intensity.
    dataset: str default: 'checkerboard'
        name of stim dataset in hdf5 file

    Return
    ------
    stim: ndarray
        3D array stimulus with the Numpy convention for dimensions.

    """
    rgb = {'r': 0, 'g': 1, 'b': 2}
    assert channel in rgb, "channel must be 'r', 'g' or 'b'."

    with h5py.File(filepath, 'r') as fstim:
        len_stim, ysize, xsize, nchannels = fstim[dataset].shape
        stim = np.zeros((len_stim, ysize, xsize), dtype=np.float64)
        fstim[dataset].read_direct(stim, np.s_[..., rgb[channel]])

    if normed:
        stim_min, stim_max = stim.min(), stim.max()
        stim = ((stim - stim_min) / ((stim_max - stim_min) / 2) - 1)

    return stim


def load_matstim(filepath, matvar='stim'):
    """
    Return checkerboard stim from .mat file.

    Read checkerboard stimulus generate by matlab and transpose
    dimesion to make compatible with numpy convencion of
    multidimentional array. The stimulus must have 4 dimension,
    nframe, yblock, xbock, nchannels.

    Parameters
    ----------
    filepath: str
        relative or full path of matlab file.
    var: str default: 'stim'
        name of stim variable in mat file

    Return
    ------
    stim: ndarray
        4D array stimulus with the Numpy convention for dimensions
        (frame,y,x,channel).

    Note
    ----
    scipy.io.loadmat read matfile in mantain the matlab axis order
    to access array, ej. shape (y,x,channel,frame) = (35,35,3,72000)
    and python should be (frame,y,x,channel) = (72000,35,35,3), for
    this reason the output file keep python format.
    https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays/
    http://scikit-image.org/docs/dev/user_guide/numpy_images.html

    """
    try:
        stim = loadmat(filepath)
        stim = stim[matvar]
        # Dimentional orden from data save in matlab v7
        (row, col, ch, pln) = (0, 1, 2, 3)
        # Scikit-image convention
        stim = np.transpose(stim, (pln, row, col, ch))
    except ValueError:
        with h5py.File(filepath, 'r') as stim_raw:
            # Dimentional orden from data save in matlab v7.3
            (pln, ch, col, row) = (0, 1, 2, 3)
            stim = np.empty(stim_raw[matvar].shape, dtype=np.uint8)
            stim_raw['stim'].read_direct(stim, np.s_[...])
            # Scikit-image convention
            stim = np.transpose(stim, (pln, row, col, ch))
    except ValueError as verror:
        verror('There are problems with {} file'.format(filepath))

    return stim


def load_stim_multi(filepath, normed=True, channel='g',
                    dataset='checkerboard'):
    """
    Get checkerboard stim from hdf5 file to run multiprocessing sta.

    Read checkerboard stimulus generate by matlab and transpose
    dimesion to make compatible with numpy convencion of
    multidimentional array. The stimulus must have 4 dimension,
    nframe, yblock, xbock, nchannels.

    Parameters
    ----------
    filepath: str
        relative or full path of hdf5 file.
    normed: bool default: True
        Normalize stimulus between -1 and 1.
    channel: {'r', 'g', 'b'}
        stimulus has 3 channels, 'r', 'g' and 'b'. Set one of these
        to retrive stim intensity.
    dataset: str default: 'checkerboard'
        name of stim dataset in hdf5 file

    Return
    ------
    stim: RawArray
        3D array stimulus with the Numpy convention for dimensions.
    stim_shape: tuple
    """
    stim_raw = load_stim(filepath, normed=normed, channel=channel,
                         dataset=dataset)
    len_stim, ysize, xsize = stim_raw.shape
    stim = RawArray('d', len_stim*ysize*xsize)
    stim_np = np.frombuffer(stim, dtype=np.float64).reshape(stim_raw.shape)
    np.copyto(stim_np, stim_raw)

    return stim, stim_raw.shape


def load_spk_txt():
    pass


def load_spk_hdf5(filepath, group, datasets=None):
    """
    Return spiketimes from hdf5 file.

    Read HDF5 file and get all spike inside a group

    Parameters
    ----------
    filepath: str
        relative or full path of hdf5 file.
    group: str
        name of the group where timestams are
    datasets: list
        list of a specifict name of datasets in the group

    Return
    ------
    spks: dic
        dictionary with timestams

    """
    spks = {}
    with h5py.File(filepath, 'r') as fspks:
        if datasets:
            for key in datasets:
                spks[key] = fspks[group+key][...]
        else:
            for key in fspks[group]:
                spks[key] = fspks[group+key][...]
    return spks
