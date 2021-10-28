import numpy as np
from pyvtna.signal import is_ascending


class OverlapMetric():
    def __init__(self, max_is_best=False):
        self.mib = max_is_best
    def __call__(self, model, data):
        return 1.0


class R2(OverlapMetric):
    def __call__(self, model, data):
        rsquared = r2(model, data)
        if self.mib:
            if not is_ascending(model):
                return -rsquared
            return rsquared


class AdjR2(OverlapMetric):
    def __call__(self, model, data, k=1):
        arsquared = adj_r2(model, data, k)
        if self.mib:
            if not is_ascending(model):
                return -arsquared
            return arsquared


class PearR(OverlapMetric):
    def __call__(self, model, data):
        r = pear_r(model, data)
        if self.mib:
            if not is_ascending(model):
                return -r
            return r


class MAD(OverlapMetric):
    def __call__(self, model, data):
        nsad = mean_abs_diff(model, data)
        if self.mib:
            return -nsad
        return nsad


class RMSD(OverlapMetric):
    def __call__(self, model, data):
        r_m_s_d = rmsd(model, data)
        if self.mib:
            return -r_m_s_d
        return r_m_s_d


def r2(model, data):
    """
    Compute the squared correlation coefficient for predicted values against data.
    
    Parameters
    ----------
    model: numpy.ndarray
        The predicted values as an one-dimensional numpy array.
    data: numpy.ndarray
        The actual values as an one-dimensional numpy array.
    Returns
    -------
    float
    """
    SStot = np.sum((data - np.mean(data))**2)
    SSres = np.sum((data - model)**2)
    return 1 - SSres / SStot


def adj_r2(model, data, k=1):
    """
    Compute the adjusted squared correlation coefficient for predicted values against data
    
    Penalizes a larger number of parameters used in the model.
    
    Parameters
    ----------
    model: numpy.ndarray
        The predicted values as an one-dimensional numpy array.
    data: numpy.ndarray
        The actual values as an one-dimensional numpy array.
    k: int, optional
        The number of parameters / features in the model.
    Returns
    -------
    float
    """    
    n = len(data)
    SStot = np.sum((data - np.mean(data))**2)
    SSres = np.sum((data - model)**2)
    R2 = 1 - SSres / SStot
    return 1-(1-R2)*(n-1)/(n-k-1)


def pear_r(model, data):
    """
    Compute the Pearson Correlation Coefficient predicted values against data
    
    Parameters
    ----------
    model: numpy.ndarray
        The predicted values as an one-dimensional numpy array.
    data: numpy.ndarray
        The actual values as an one-dimensional numpy array.
    Returns
    -------
    float
    """    
    num = np.sum((model - np.mean(model)) * (data - np.mean(data)))
    denom = np.sqrt(np.sum((model - np.mean(model))**2))*np.sqrt(np.sum((data - np.mean(data))**2))
    return num / denom


def mean_abs_diff(model, data):
    """
    Compute the normalized sum of the absolute differences between predicted values against data

    Parameters
    ----------
    model: numpy.ndarray
        The predicted values as an one-dimensional numpy array.
    data: numpy.ndarray
        The actual values as an one-dimensional numpy array.
    Returns
    -------
    float
    """
    return np.abs((model - data)).sum() / len(model)

def rmsd(model, data):
    """
    Compute the root mean squared deviation between predicted values against data

    Parameters
    ----------
    model: numpy.ndarray
        The predicted values as an one-dimensional numpy array.
    data: numpy.ndarray
        The actual values as an one-dimensional numpy array.
    Returns
    -------
    float
    """
    return np.sqrt(((model - data)**2).sum() / len(model))