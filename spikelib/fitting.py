"""Set of function to fit a raw data."""
import numpy as np
from lmfit import Minimizer, Parameters


def two_cascades(params, time):
    """
    Model for temporal integration of sta.

    This function return the model of the difference of two cascades
    of low-pass filters for temporal integration of spike triggered
    average (sta). It was propouse by Chichilnisky and Kalmar in
    'Functional Asymmetries In ON and OFF Ganglion Cells Of Primate
    Retina'.

    function

    Parameters
    ----------
    params : dict or lmfit.Params.params
        The dictionary has 5 element for each parameter of model,
        'amp1', 'amp2', 'tau1', 'tau2', 'n'
    time : ndarray
        Array with the time to compute the model.

    Return
    ------
    model: ndarray
        return values of the function for each element in time array

    """
    amp1 = params['amp1']
    amp2 = params['amp2']
    tau1 = params['tau1']
    tau2 = params['tau2']
    order = params['n']
    gauss1 = amp1 * (time/tau1)**order * np.exp(-order * (time/tau1-1))
    gauss2 = amp2 * (time/tau2)**order * np.exp(-order * (time/tau2-1))
    model = gauss1 - gauss2
    return model


def two_cascades_min(params, x, raw_data):
    """Difference of two cascades to minimize."""
    to_minimize = two_cascades(params, x) - raw_data
    return to_minimize


def fit_temp_sta(temporal_sta, time, fit_time, tau1=None, tau2=None, amp1=None,
                 amp2=None, min_time=None, max_time=None, min_amp=-1,
                 max_amp=1, max_n=20):
    """
    Fit the temporal integration of the sta.

    Use the difference of two cascades of low-pass filters to fit
    the raw temporal integration of STA. It uses the time before
    of the spike to compute the fitting.

    Parameters
    ----------
    temporal_sta: ndarray
        array with the raw temporal integration of the sta.
    time: ndarray
        array with the time of the raw temporal integration.
    fit_time: ndarray
        array with the time of fitting curve.
    tau1: float default:None
        estimated time for positive peak of temporal integration
    tau2: float default:None
        estimated time for negative peak of temporal integration
    amp1: float default:None
        estimated amplitude for positive peak of temporal integration
    amp2: float default:None
        estimated amplitude for negative peak of temporal integration
    min_time: float default:None
        minimum time to fit tau1 or tau2
    max_time: float default:None
        maximum time to fit tau1 or tau2
    min_amp: float default:-1
        minimum amplitude to fit amp1 or amp2
    max_amp: float default:1
        minimum amplitude to fit amp1 or amp2
    max_n=: float default:20
        maximum order of a model to fit

    Return
    ------
    fit_parameters: lmfit.Params.params
        parameters of the fitting for two_cascades model
    fit_temp: ndarray
        array with the values of the fitting using fit_time

    """
    tau1 = tau1 if tau1 else time[temporal_sta.argmax()]
    tau2 = tau2 if tau2 else time[temporal_sta.argmin()]
    amp1 = amp1 if amp1 else np.abs(temporal_sta.max())
    amp2 = amp2 if amp2 else np.abs(temporal_sta.min())
    min_time = min_time if min_time else time[1]
    max_time = max_time if max_time else time[-2]

    params_fit = Parameters()
    params_fit.add('amp1', value=amp1, min=min_amp, max=max_amp)
    params_fit.add('amp2', value=amp2, min=min_amp, max=max_amp)
    params_fit.add('tau1', value=tau1, min=min_time, max=max_time)
    params_fit.add('tau2', value=tau2, min=min_time, max=max_time)
    params_fit.add('n', value=3, max=max_n)

    minner = Minimizer(two_cascades_min, params_fit,
                       fcn_args=(time, temporal_sta))
    try:
        result = minner.minimize(method='Nelder')
        fit_parameters = result.params
        fit_temp = two_cascades(fit_parameters, fit_time)
    except ValueError:
        try:
            result = minner.minimize()
            fit_parameters = result.params
            fit_temp = two_cascades(fit_parameters, fit_time)
        except ValueError:
            for key in params_fit:
                params_fit[key].set(value=0)
            fit_parameters = params_fit
            fit_temp = np.full_like(fit_time, np.nan)

    return fit_parameters, fit_temp
