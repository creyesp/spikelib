"""Set of test for fitting module."""
import numpy as np

# import context
import spikelib.fitting as fitt


def test_fit_temp_sta():
    """Test for fitting temporal integration of sta."""
    time_raw = np.linspace(-0.5, 0, 30)
    time_fit = np.linspace(-0.5, 0, 30)
    params = {'amp1': 0.5, 'amp2': 0.9, 'tau1': -0.2, 'tau2': -0.1, 'n': 7}

    np.random.seed(0)
    sta_noise = (np.random.rand(30,)-0.5)/5
    test_model = fitt.two_cascades(params, time_raw)
    test_raw = test_model+sta_noise
    params_fit, test_fit = fitt.fit_temp_sta(test_raw, time_raw, time_fit)

    assert np.all(np.abs(test_fit - test_model) <= 0.05), \
        'Difference between model and fitting more than 5% by negative time'

    time_raw = np.linspace(0, 0.5, 30)
    time_fit = np.linspace(0, 0.5, 30)
    params = {'amp1': 0.5, 'amp2': 0.9, 'tau1': 0.2, 'tau2': 0.1, 'n': 7}

    np.random.seed(0)
    sta_noise = (np.random.rand(30,)-0.5)/5
    test_model = fitt.two_cascades(params, time_raw)
    test_raw = test_model+sta_noise
    params_fit, test_fit = fitt.fit_temp_sta(test_raw, time_raw, time_fit)

    assert np.all(np.abs(test_fit - test_model) <= 0.05), \
        'Difference between model and fitting more than 5% by positive time'
