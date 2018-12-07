import sys
import os
sys.path.append(os.path.realpath('..'))

import numpy as np
import matplotlib.pyplot as plt

from spikelib.analysis import flash


def plot_flash(type_fit, char_fit):
    ccolor = {0: 'k', 1: 'r', 2: 'b', 3: 'g', 4: 'k'}

    latency_on = char_fit[0]
    latency_off = char_fit[1]
    dacay_on = char_fit[3]
    decay_off = char_fit[4]
    start_on, end_on, start_off, end_off = bound
    max_amp = response.max()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    fig.suptitle(key)
    ax1.plot(time_resp, response, c=ccolor[type_fit])
    ax1.step(nbins[1:], nc)
    ax1.vlines(start_on+latency_on, 0, max_amp, alpha=0.5)
    ax1.vlines(start_on+latency_on+dacay_on, 0, max_amp, alpha=0.5)
    ax1.vlines(start_off+latency_off, 0, max_amp, alpha=0.5)
    ax1.vlines(start_off+latency_off+decay_off, 0, max_amp, alpha=0.5)
    ax1.set_xlim([start_on, end_off])

    for krep, ktrial in enumerate(trials_flash):
        ax2.scatter(ktrial, krep*np.ones_like(ktrial),
                    marker='|', c='k', alpha=0.5)
    ax2.vlines(start_on+latency_on, 0, max_amp, alpha=0.5)
    ax2.vlines(start_on+latency_on+dacay_on, 0, max_amp, alpha=0.5)
    ax2.vlines(start_off+latency_off, 0, max_amp, alpha=0.5)
    ax2.vlines(start_off+latency_off+decay_off, 0, max_amp, alpha=0.5)
    ax2.set_xlim([start_on, end_off])
    plt.show()


if __name__ == "__main__":
    psth_bin = 0.02  # in sec

    on_dur = 3.0124  # seconds 60248 3.0124
    trans_time = 0.0  # seconds
    off_dur = 3.0124  # seconds

    bound = (-on_dur, trans_time, trans_time, off_dur)  # seconds
    resp_thr = 1/3.0  # Threshold psth [spike in a bin]
    bias_thr = 0.65  # thr to classify into on, off, onoff
    ri_thr = 0.3  # Threshold for Response Index
    fpeak_thr = 0.5  # % of peak response to find peaks
    fpeak_min_dist = 10  # samples
    ri_span = 0.1  # seg
    sust_time = 0.4  # seg
    decrease_factor = np.e
    samplerate = 20000.0  # samples of record

    # Load data
    key = 'temp252'
    spikes = np.loadtxt('data/{}.txt'.format(key))
    trials_time = np.loadtxt('data/{}.txt'.format('flash_trials'))/samplerate
    start_trials = trials_time[:, 0]
    end_trials = trials_time[:, 1]
    trials_flash = flash.get_trials(spikes, start_trials,
                                    end_trials, offset=on_dur)
    spks_flash = flash.flatten_trials(trials_flash)
    ntrails = len(trials_flash)
    psth_bins = int((on_dur+off_dur)/psth_bin)

    # Response
    [nc, nbins] = np.histogram(spks_flash, bins=psth_bins,
                               range=(bound[0], bound[-1]))
    nc = nc/float(ntrails)
    time_resp = np.linspace(bound[0], bound[-1], int((off_dur+on_dur)*1000))
    response = flash.est_pdf(trials_flash, time_resp, bandwidth=psth_bin,
                             norm_factor=nc.max())

    kws = {'bias_thr': bias_thr, 'ri_thr': ri_thr, 'fpeak_thr': fpeak_thr,
           'fpeak_min_dist': 2, 'ri_span': ri_span,
           'sust_time': sust_time, 'decrease_factor': decrease_factor}
    type_psth, char_psth = flash.temporal_characterization(
        nc, nbins[1:], bound, resp_thr, **kws)
    kws_fit = {'bias_thr': bias_thr, 'ri_thr': ri_thr, 'fpeak_thr': fpeak_thr,
               'fpeak_min_dist': fpeak_min_dist, 'ri_span': ri_span,
               'sust_time': sust_time, 'decrease_factor': decrease_factor}
    type_fit, char_fit = flash.temporal_characterization(
        response, time_resp, bound, resp_thr, **kws_fit)

    print('Result using PSTH with bin={}'.format(psth_bin))
    print(type_psth, char_psth)
    print('Result using PDF with bin={}'.format(psth_bin))
    print(type_fit, char_fit)

    plot_flash(type_fit, char_fit)
