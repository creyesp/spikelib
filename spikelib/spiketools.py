"""Set of functions to work with spike times."""
import numpy as np
from peakutils import indexes as pindexes
from sklearn.neighbors import KernelDensity


def chunk_spikes(spikes, start, end):
    """
    Get a subset of spike between start and end time

    Parameters
    ----------
    spikes: ndarray
        array with timestams of a cell
    start: float
        time of start the chunk
    end: float
        time of end the chunk

    Returns
    -------
    chunk: ndarray
        subset of spikes

    """
    chunk = spikes[(spikes >= start)*(spikes <= end)]
    return chunk


def get_trials(spikes, start_trials, end_trials, offset=0):
    """
    Get a list of spiketime for each trails.

    Search spike times in set of period of time(s) defined in
    start_trials and end_trails and return a list of spiketimes
    for each trails and set the start time to 0 for each trail.

    Parameters
    ----------
    spikes : ndarray
        Array of all spike times.
    start_trials : ndarray
        Set of start time for each trail.
    end_trials: ndarray
        Set of end time for each trail.
    offset : float
        offset allow move the reference of start time.

    Return
    ------
    list
        List of array of all spikes for each trial.

    """
    msg = 'start and end time must have equal length'

    assert len(start_trials) == len(end_trials), msg

    start_time = start_trials[0]
    end_time = end_trials[-1]
    spks = chunk_spikes(spikes, start_time, end_time)
    spks_trials = []
    for (kstart, kend) in zip(start_trials, end_trials):
        filter_trial = (spks >= kstart)*(spks < kend)
        spk_trial = spks[filter_trial] - kstart - offset
        spks_trials.append(spk_trial)
    return spks_trials


def flatten_trials(trials):
    """Put spike times from different trial in a 1D array."""
    spks = []
    for ktrial in trials:
        spks.extend(ktrial)
    return np.array(spks)


def est_pdf(trails, time, bandwidth=0.02, norm_factor=1):
    """
    Estimate the probability density function from spike.

    This function estimate the pdf of a cell to get a instantaneous
    firing rates.

    Parameters
    ----------
    spikes : list
        List with all spike times in each trial
    time : ndarray
        Array of time points where pdf will be estimate.

    Return
    ------
    est_pdf : ndarray

    """
    try:
        spiketimes = flatten_trials(trails)
        spiketimes = spiketimes[:, np.newaxis]
        time = time[:, np.newaxis]
        kde = KernelDensity(
            kernel='epanechnikov', bandwidth=bandwidth).fit(spiketimes)
        est_pdf = np.exp(kde.score_samples(time))
        est_pdf = (est_pdf/est_pdf.max())*norm_factor
    except ValueError:
        est_pdf = np.zeros_like(time)
    return est_pdf.flatten()


def sustain_index(response):
    """
    Get the sustained index.

    Sustained Index (SI) measure if the response is transient or
    sustained, where 0 is completly sustained and 1 is completly
    transient.

    Parameters
    ----------
    response : ndarray
        PSTH or estimated response array of a cell in a specific
        estimulation.

    Return
    ----------
    sust_index : float
        index between 0 and 1.

    """
    response_max = response.max()
    response_mean = response.mean()
    try:
        sust_index = (response_max-response_mean)/(response_max+response_mean)
    except ZeroDivisionError:
        sust_index = np.nan
    return sust_index


def bias_index(fr_max_on, fr_max_off, thr=0.65):
    """
    Get the bias index from a On Off response.

    The bias index compare the response to 2 differente stimuli, ON
    and OFF, and classify between, ON, OFF and ONOFF response.
    .. math::
        bias_index = \\frac{ON_{max}-OFF_{max}}{ON_{max}+OFF_{max}}

    Parameters
    ----------
    fr_max_on : float
        firing rate maximum for On response
    fr_max_off : float
        firing rate maximum for Off response
    thr : float
        threshold to split On, Off and On-Off groups

    Return
    ----------
    bias_index : float
        number between -1 and 1.
    response_type : int
        clasification for type of response
            0: null response
            1: On response
            2: Off response
            3: On-Off response

    """
    try:
        bias_index = (fr_max_on-fr_max_off)/float(fr_max_on+fr_max_off)
        if bias_index > thr:
            response_type = 1
        elif bias_index < -thr:
            response_type = 2
        else:
            response_type = 3
    except ZeroDivisionError:
        response_type = 0
        bias_index = np.nan

    return (bias_index, response_type)


def response_index(response, prev_response, ri_span, max_resp=None):
    """Get the response index.

    The response index compare the response to a stimulus against
    previous response to the stimulus. It measure allow cuantify
    how much to change the response to a stimulus.
    .. math::
        resp_index = \frac{resp_{max}-avg_{prev_resp}}\
    {resp_{max}+avg_{prev_resp}}

    Parameters
    ----------
    response : ndarray
        response for analysis
    prev_response : ndarray
        previous response for analysis
    ri_span : int
        number of de points to analize in response and prev_resp
    max_resp : float
        peak of response

    Return
    ----------
    resp_index : float
        value between  0 to 1, where 0 mean that the response don't
        change to the stimulus and 1 mean the previous response was
        0 and response was max.

    """
    if not max_resp:
        max_resp = np.amax(response)
    avg_resp = response[:ri_span].mean()
    avg_prev_resp = prev_response[-ri_span:].mean()
    try:
        resp_index = (max_resp-avg_prev_resp)/(max_resp+avg_resp)
    except ZeroDivisionError:
        resp_index = np.nan

    return resp_index


def decay_time(response, time, peaktime, max_resp, decrease_factor=np.e):
    """
    Get the time of the response dalay to dacay.

    Dacay time is the time that the response take to dacay $n$
    factor to the max response.

    Parameters
    ----------
    response : ndarray
        psth or estimated firingrate
    time : ndarray
        time of response
    peaktime : float
        time where is the max response
    max_resp : int
        value of the max response
    decrease_factor : float
        number of time that maximum response should dacay to
        get decay_time

    Return
    ----------
    resp_index : float
        value between  0 to 1, where 0 mean that the response don't
        change to the stimulus and 1 mean the previous response was
        0 and response was max.

    """
    fiter_decay = (time > peaktime) & (response < max_resp/decrease_factor)
    decay = time[fiter_decay]
    dacay_time = decay[0] if decay.any() else time[-1]
    dacay_time = dacay_time - peaktime
    return dacay_time


def get_features_flash(response, time_resp, bound, resp_thr=0.3,
                       bias_thr=0.65, ri_thr=0.3, fpeak_thr=0.3,
                       fpeak_min_dist=10, ri_span=0.1, sust_time=0.4,
                       decrease_factor=np.e
                       ):
    """
    Get temporal characteristic of flash response.

    Parameter
    ------------
    response : ndarray
        psth or estimated firingrate
    time_resp, array
        time of response in seg
    bound : tuple
        (start_time_on, end_time_on, start_time_off, end_time_off)
    resp_thr : float default=0.3
        threshold to define if response is valid
    bias_thr : float default=0.65
        threshold for bias index
    ri_thr : float default=0.3
        threshold for response index
    fpeak_thr : float default=0.3
        threshold for find peak response
    fpeak_min_dist : int default=10
        number of point minimum to detect peak response
    ri_span=0.1 : float default=0.1
        windows time to compute response index in seg
    sust_time : float default=0.4
        windows time to compute sutained index in seg
    decrease_factor : float default=np.e
        feaactor to get decrease time

    Return
    ----------
    flash_type : int
        Flash clasification, 0, 1, 2 or 3 that represent Null, ON,
        OFF and ONOFF.
    flash_char : ndarray
        Array with the follow paramiters: [latency_on, latency_off,
        bias_idx, decay_on, decay_off, resp_index_on,
        resp_index_off, sust_index_on, sust_index_off, fr_max_on,
        fr_max_off]
    """
    response = np.asarray(response)
    (start_on, end_on, start_off, end_off) = bound  # seconds

    filter_on_time = (start_on <= time_resp) & (time_resp < end_on)
    filter_off_time = (start_off <= time_resp) & (time_resp < end_off)
    time_resp_on = time_resp[filter_on_time]
    time_resp_off = time_resp[filter_off_time]
    ri_span_samples = int(ri_span/(time_resp[1]-time_resp[0]))

    filter_sust_on = time_resp_on <= (time_resp_on[0]+sust_time)
    filter_sust_off = time_resp_off <= (time_resp_off[0]+sust_time)

    if np.amax(response) > resp_thr:

        resp_on = response[filter_on_time]
        resp_off = response[filter_off_time]
        if (resp_on.max() > resp_thr):
            peak_idx_on = pindexes(resp_on, thres=fpeak_thr,
                                   min_dist=fpeak_min_dist)[0]
            fr_max_on = resp_on[peak_idx_on]
            peaktime_on = time_resp_on[peak_idx_on]
            latency_on = peaktime_on - start_on
            decay_on = decay_time(resp_on, time_resp_on, peaktime_on,
                                  fr_max_on, decrease_factor)
            resp_index_on = response_index(resp_on, resp_off, fr_max_on,
                                           ri_span_samples)
            sust_index_on = sustain_index(resp_on[filter_sust_on])
        else:
            peak_idx_on = 0
            fr_max_on = 0
            peaktime_on = 0
            latency_on = 0
            decay_on = 0
            resp_index_on = 0
            sust_index_on = 0

        if (resp_off.max() > resp_thr):
            peak_idx_off = pindexes(resp_off, thres=fpeak_thr,
                                    min_dist=fpeak_min_dist)[0]
            fr_max_off = resp_off[peak_idx_off]
            peaktime_off = time_resp_off[peak_idx_off]
            latency_off = peaktime_off - start_off
            decay_off = decay_time(resp_off, time_resp_off, peaktime_off,
                                   fr_max_off, decrease_factor)
            resp_index_off = response_index(resp_off, resp_on, fr_max_off,
                                            ri_span_samples)
            sust_index_off = sustain_index(resp_off[filter_sust_off])
        else:
            peak_idx_off = 0
            fr_max_off = 0
            peaktime_off = 0
            latency_off = 0
            decay_off = 0
            resp_index_off = 0
            sust_index_off = 0

        if (resp_index_on < ri_thr) | (fr_max_on < resp_thr):
            peak_idx_on = 0
            fr_max_on = 0
            peaktime_on = 0
            latency_on = 0
            decay_on = 0
            sust_index_on = 0
            resp_index_on = 0

        if (resp_index_off < ri_thr) | (fr_max_off < resp_thr):
            peak_idx_off = 0
            fr_max_off = 0
            peaktime_off = 0
            latency_off = 0
            decay_off = 0
            sust_index_off = 0
            resp_index_off = 0

        bias_idx, flash_type = bias_index(fr_max_on, fr_max_off, bias_thr)
        flash_char = np.asarray([latency_on,
                                 latency_off,
                                 bias_idx,
                                 decay_on,
                                 decay_off,
                                 resp_index_on,
                                 resp_index_off,
                                 sust_index_on,
                                 sust_index_off,
                                 fr_max_on,
                                 fr_max_off,
                                 ])
        if not flash_type:
            flash_char = np.full((11,), np.nan)
    else:
        flash_type = 0
        flash_char = flash_char = np.full((11,), np.nan)

    return (flash_type, flash_char)
