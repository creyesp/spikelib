"""Set of tools for visualization."""
import numpy as np
import matplotlib.pyplot as plt


def raster(trials, ax=None, **kwargs):
    """
    Draw raster for a cell.

    Parameters
    ----------
    trials: list of ndarray
        list with spiketimes for each trial
    **kwargs:
        kwargs of axis of matplotlib

    Returns
    -------
    fig: figure object
        current figure of the plot
    ax: axis object
        axis with raster

    """
    if not ax:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    kws = {'marker': '|', 'c': 'k', 'alpha': 0.5}
    kws.update(kwargs)
    for kidx, ktrial in enumerate(trials):
        ax.scatter(ktrial, np.ones(len(ktrial))*kidx, **kws)

    return fig, ax


def psth(spks, ax=None, **kwargs):
    """
    Draw psth for spikes.

    Parameters
    ----------
    spks: ndarray
        list with spiketimes for each trial
    **kwargs:
        kwargs of axis of matplotlib

    Returns
    -------
    fig: figure object
        current figure of the plot
    ax: axis object
        axis with psth

    """
    assert spks.ndim == 1, 'spks must be a 1D array'

    if not ax:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    ax.hist(spks, **kwargs)

    return fig, ax


def plot_sta(sta_array, name=''):
    """
    Get a plot with all frame of the STA.

    Parameters
    ----------
    sta_array: ndarray
        STA array
    name: str
        name of unit

    Returns
    -------
    fig: figure object
        current figure of the plot
    ax: axis object
        axis with psth

    """
    nframes = sta_array.shape[0]
    ncol = 6
    nrow = nframes//ncol+1 if nframes % ncol else nframes//ncol
    max_c = (np.abs(sta_array)).max()
    fig, ax = plt.subplots(nrow, ncol,
                           sharex=True, sharey=True,
                           figsize=(ncol*1.5, nrow*1.5),
                           )
    axf = ax.flatten()
    for kidx, kframe in enumerate(sta_array):
        img = axf[kidx].pcolor(kframe, vmin=-max_c, vmax=max_c, cmap='RdBu_r')
        axf[kidx].set_title('frame {}'.format(nframes-kidx-1), fontsize=6)
        axf[kidx].set_aspect(1)
    fig.colorbar(img, ax=ax, orientation='vertical', fraction=.01,
                 label='Range of stimulu [-1,1]')
    fig.suptitle(name)
    return (fig, ax)
