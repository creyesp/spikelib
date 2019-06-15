"""spikelib utilities."""
import os

import numpy as np


def clean_directory(inputfolder):
    """
    Delete all files and drectories in input directory.

    If the directory doesn't exist then this directory is create.

    Parameters
    ----------
    inputfolder : str
        path of directory

    Examples
    --------
    >>> from mealib.utils import clean_directory
    >>> clean_directory('../myinputfolder')

    """
    if not os.path.exists(inputfolder):
        try:
            os.makedirs(inputfolder)
        except NotADirectoryError as err:
            print('Unable to create folder ' + inputfolder)
            raise err
    else:
        if os.listdir(inputfolder) != []:
            for files in os.listdir(inputfolder):
                os.remove(inputfolder+files)


def check_directory(inputfolder):
    """
    Check if exist the directory, else make a new folder.

    If the directory doesn't exist then this directory is create.

    Parameters
    ----------
    inputfolder : str
        path of directory

    Examples
    --------
    >>> checkDirectory('../myFolder')

    """
    if not os.path.exists(inputfolder):
        try:
            os.makedirs(inputfolder)
        except NotADirectoryError as err:
            print('Unable to create folder ' + inputfolder)
            raise err


def check_groups(fdata, groups):
    """
    Review if a list of groups exist in a hdf file.

    Parameters
    ----------
    fdata : object
        h5py object to check if a list of group exist
    groups : list of str
        list of group name to check

    """
    for kgroup in groups:
        if kgroup not in fdata:
            fdata.create_group(kgroup)


def datasets_to_array(fgroup):
    """
    Transform all datasets in a group to an array.

    Parameters
    ----------
    fgroup : object
        h5py object to retrive all dataset inside of it

    Returns
    -------
    ds_array : array_like
        NxM array, where N if the number of dataset and M is the
        lenght of each dataset
    names : list of str
        list of names for each row in ds_array

    Examples
    --------
    >>> with h5py.File(fdata) as data:
    >>>    array, keys = datasets_to_array(data[group])
    >>> array.shape
    (889,7)

    """
    ds_array = []
    names = []
    for key in fgroup:
        ds_array.append(fgroup[key][...])
        names.append(key)
    ds_array = np.array(ds_array)

    return ds_array, names
