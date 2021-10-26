import numpy as np


def R2(model, data):
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

def adj_R2(model, data, k=1):
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
