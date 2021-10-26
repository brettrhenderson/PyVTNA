import numpy as np
from scipy.interpolate import interp1d

def interp_neg(t, s):
    """
    Remove all negative points in a signal and interpolate between the surrounding points.
    
    Preserves number of points. Uses linear interpolation.
    
    Parameters
    ----------
    t: numpy.ndarray
        Time sequence for the signal.
    s: numpy.ndarray
        The signal to interpolate.
    Returns
    -------
    None
    """
    s[s < 0] = np.interp(t[s < 0], t[s >= 0], s[s >= 0], left=0, right=0)
    
def drop_neg(t, ss):
    """
    Remove all negative points in a signal.
    
    Number of points is reduced.
    
    Parameters
    ----------
    t: numpy.ndarray
        Time sequence for the signal.
    ss: numpy.ndarray
        An m x n array of n different signals at m time steps. Negatives are removed from each signal.
    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
    """
    t = t[ss.min(axis=1)>=0]
    ss = ss[ss.min(axis=1)>=0]
    return t, ss
    
def interp_neg_comp(t, s, method='linear'):
    """
    Remove all negative points in a signal and interpolate between the surrounding points.
    
    Preserves number of points. Uses linear, nearest neighbor, 
    linear, quadratic, or cubic spline interpolation.
    
    Parameters
    ----------
    t: numpy.ndarray
        Time sequence for the signal.
    s: numpy.ndarray
        The signal to interpolate.
    method: string, optional {‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’}
        The interpolation method to use. 
    Returns
    -------
    None
    """
    f = interp1d(t[s >= 0], s[s >= 0], fill_value=0, kind=method)
    s[s < 0] = f(t[s < 0])
    
def replace_neg(s, val=0):
    """
    Replace all negative points in a signal with a set value.
    
    Number of points is preserved.
    
    Parameters
    ----------
    s: numpy.ndarray
        A signal in which to replace negative values
    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
    """
    s[s < 0] = val

def x_rolling(x, win=3):
    """
    Return the x_values for the data returned by taking a rolling mean using nan()
    
    Parameters
    ----------
    x: numpy.ndarray
        The original array of x-values for the series of data that is averaged.
    win: int
        The window size for taking the average (window is centered about the resulting mean values.
    Returns
    -------
    numpy.ndarray
    """    
    miss = (win-1) // 2
    rem = (win-1) % 2
    if miss != 0:
        return x[miss+rem:-miss]
    else:
        return x[miss+rem:]

# Easy - Rolling Average
def rolling_mean(arr, win=3):
    """
    Return the rolling average of an array with set window length and uniform weighting
    
    Parameters
    ----------
    arr: numpy.ndarray
        The array that to be averaged.
    win: int
        The window size for taking the average (window is centered about the resulting mean values.
    Returns
    -------
    numpy.ndarray    
    """    
    if win <= 0:
        return arr
    roll = np.cumsum(arr, dtype=float)
    roll[win:] = roll[win:] - roll[:-win]
    return roll[win - 1:] / win

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size. 
    (https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html)
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if window_len & 1:
        window_len += 1
    
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    #s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if is_ascending(x):
        s = np.r_[x[0]-x[window_len-1:0:-1], x, x[-1] - (x[-1] - x[-2:-window_len-1:-1])]
    else:
        s = np.r_[x[0] + x[0] - x[window_len-1:0:-1], x, x[-1] + (x[-1] - x[-2:-window_len-1:-1])]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[int((window_len/2-1)):-int((window_len/2))]

def dv_dx(vs, xs=None):
    """
    Take the derivative of a 1D array vs with spacing xs
    
    Parameters
    ----------
    vs: numpy.ndarray
        The function to take a derivative of, as a 1D array
    xs: numpy.ndarray or float
        The spacing between the function values, either as a 1D array or a float. 
        If a float, uniform spacing is assumed
    Returns
    -------
    numpy.ndarray
    """
    dv = vs[1:] - vs[:-1]
    if xs is not None:
        dx = xs[1:] - xs[:-1]
    else:
        dx = np.ones(len(dv))
    return dv/dx

def signaltonoise(sig, axis=0, ddof=0):
    """
    Return the signal to noise ratio of a signal s
    
    Parameters
    ----------
    sig: numpy.ndarray
        The signal on which to evaluate the signal to noise ratio.
    axis: int
        The axis along which to evaluate the signal-to-noise ratio.
    ddof: int
        Delta degrees of freedom when calculating the standard deviation
        via numpy.std. The number of degrees of freedom used for N values is N-ddof.
    Returns
    -------
    float
    """
    numpy.ndarray
    a = np.asanyarray(sig)
    m = sig.mean(axis)
    sd = sig.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def rangeSNR(sig):
    """
    Return the ratio of the signal range to its standard deviation.
    
    Parameters
    ----------
    sig: numpy.ndarray
        The signal on which to evaluate the ratio.
    Returns
    -------
    float
    """
    return (np.max(sig) - np.min(sig)) / np.std(sig)
    
    
def derivSNR(sig):
    """
    Return the ratio of maximum magnitude of the signal derivative 
    to the standard deviation of the signal derivative
    
    Parameters
    ----------
    sig: numpy.ndarray
        The signal on which to evaluate the ratio.
    axis: int
        The axis along which to evaluate the signal-to-noise ratio.
    Returns
    -------
    float
    """
    return np.max(np.abs(dv_dx(sig))) / np.std(dv_dx(sig))
    

def get_best_win(sig, minwin=5, maxwin=100, mode='derivative'):
    """
    Chooses a smoothing window size for maximizing the signal-to-noise ratio.
    
    Parameters
    ----------
    
    arr : numpy.ndarray
          The one-dimensional data to be smoothed.
          
    minwin : int, optional
             The minimum window size to consider for smoothing.
             
    maxwin : int, optional
             The maximum window size to consider for smoothing.
             
    mode : {'derivative', 'standard', 'range'}
           Specifies the metric used to calculate the signal to noise
           ratio. In 'derivative' mode, SNR is the ratio of the maximum
           value of the absolute value of the derivative of the signal
           to the standard deviation of that derivative. In 'standard'
           mode, the SNR is calculated as the normal ratio of the signal
           mean to the signal standard deviation.  In 'range' mode, SNR
           is calculated as the ratio of the signal range to signal 
           standard deviation.
    
    Returns
    -------
    
    win : int
          The optimal window size for smoothing.
             
    """
    
    # load selected SNR method
    if mode == 'standard':
        SNR = signaltonoise
    elif mode == 'range':
        SNR = rangeSNR
    else:
        SNR = derivSNR
    
    if maxwin <= minwin:
        return min(minwin, maxwin)
    
    snr = []
    wins = np.arange(minwin,maxwin, 1)
    for win in wins:
        snr.append(SNR(rolling_mean(sig, win=win)))
    return wins[np.argmax(np.array(snr))]


def is_ascending(sig):
    """
    Decide whether a signal is generally increasing.
    
    Return True if the mean of the second half of the signal is larger
    than the mean of the first half.
    
    Parameters
    ----------
    sig: numpy.ndarray
        The signal to process
    Returns
    -------
    bool
    """
    return np.mean(sig[:int(len(sig)/2)]) < np.mean(sig[int(len(sig)/2):])


def shift_zero(sig, time, window=1):
    """
    Remove all points before actual signal starts.
    
    Find the maximum of the derivative of the signal, 
    since the signal begins with a jump from the baseline.
    Then remove all data points from before this maximum.
    Works best with a non-noisy signal, so we pass
    the signal through a low-pass filter first.
    
    Parameters
    ----------
    sig: numpy.ndarray
        The signal to shift
    time: numpy.ndarray
        The time array for the signal
    window: int, optional
        The window size to be used for smoothing the signal
    Returns
    -------
    numpy.ndarray
    """
    try:
        smoothed = rolling_mean(sig, win=window)
        t_smooth = x_rolling(time, win=window)
    except:
        smoothed = sig
        t_smooth = time
    start_idx = np.argmax(np.abs(dv_dx(smoothed, t_smooth)))   #+ 1 + int(np.ceil(window/2))
    return sig[start_idx:], time[start_idx:] - time[start_idx]


def filter_shift(sig, time, win=1):
    """
    Like shift_zero but first smooths, then shifts with a correction for the smoothing.
    
    The smoothing will shift the zero to the right by half of a window size.
    Therefore, the zero is shifted left by half of the window size here
    
    Parameters
    ----------
    sig: numpy.ndarray
        The signal to shift
    time: numpy.ndarray
        The time array for the signal
    window: int, optional
        The window size to be used for smoothing the signal
    Returns
    -------
    numpy.ndarray
    """
    shift_sig, shift_t = shift_zero(sig, time, window=win)
    try:
        shift_sig = smooth(shift_sig, window_len=win, window='blackman')
        return shift_sig, shift_t - shift_t[0]
        # return rolling_mean(shift_sig, win=window), x_rolling(shift_t, win=window) - x_rolling(shift_t, win=window)[0]
    except:
        return shift_sig, shift_t - shift_t[0]

    
def shift_zero_spear(sig, t):
    """
    Remove all points before actual signal starts.
    
    Compare signal to a test signal that has a flat
    start and then is monotonically increasing if the
    actual signal is increasing or decreasing if the 
    actual signal is decreasing.  The actual signal is 
    translated and then the spearman rank correlation 
    computed for it and the test signal.  We align the
    signal with the test signal according to the 
    translation that produces the highest correlation and
    then remove every point before the signal monotonic
    increase or decrease begins.
    
    Parameters
    ----------
    sig: numpy.ndarray
        The signal to shift
    t: numpy.ndarray
        The time array for the signal

    Returns
    -------
    numpy.ndarray
    """
    
    # 0. store the best correlation, and the corresponding signal and time
    best_r = 0
    best_t = t
    rs = []

    # 1. Zero the signal time
    t = t - t[0]

    # 2. Generate a ramp reference function
    interqlen = get_percentile_idx(sig, 75) - get_percentile_idx(sig, 25)
    if is_ascending(sig):
        ramp = np.concatenate([np.repeat(0., len(sig)), 
                               np.linspace(0., 1., interqlen * 2)])
    else:
        ramp = np.concatenate([np.repeat(0., len(sig)), 
                               np.linspace(0., -1., interqlen * 2)])
    t_ramp = np.linspace(0., len(ramp) / len(t) * t[-1], len(ramp))

    # 3. generate an array of possible translations of sig
    trans = t[1:]
    for tran in trans:
        # 4. translate the signal to the right by one point.
        t_shift = t + tran

        # 5. Find overlap between signal and ramp
        min_t = np.max([np.min(t_ramp), np.min(t_shift)])
        max_t = np.min([np.max(t_ramp), np.max(t_shift)])
        t_over = t_shift[(t_shift >= min_t) & (t_shift <= max_t)]
        s_over = sig[(t_shift >= min_t) & (t_shift <= max_t)]

        # 6. resample the ramp to match the sampling of sig
        f = interp1d(t_ramp, ramp)
        ramp_over = f(t_over)

        # 7. Compute spearman correlation
        r = spearmanr(s_over, ramp_over)[0]
        rs.append(r)

        # 8. Check if new best r
        if abs(r) > best_r:
            best_r = r
            best_t = t_shift

    # 9. Remove all points in time before the ramp begins
    best_t_trunc = best_t[best_t > t_ramp[len(t)]]
    best_sig_trunc = sig[best_t > t_ramp[len(t)]]

    # 10. Re-zero the signal time
    return best_sig_trunc, best_t_trunc - best_t_trunc[0]
                                                                     
                              
def filter_shift_spear(sig, time, window=5,):
    """
    Remove all points before actual signal starts.
    
    The same as shift_zero_spear(), but a rolling mean is taken
    to smooth the signal before shift.
    
    Parameters
    ----------
    sig: numpy.ndarray
        The signal to shift
    time: numpy.ndarray
        The time array for the signal
    window: int, optional
        The window size to be used for smoothing the signal

    Returns
    -------
    numpy.ndarray
    """
    # 1. Filter 
    smoothed = rolling_mean(sig, time=time, win=window)
    t_smoothed = x_rolling(time.copy(), win=window)
    # 2. shift
    return shift_zero_spear(smoothed, t_smoothed)
    
    
def closest_idx(sig, val):
    """
    Get the index of the value in a signal sig that is closest to val.
    
    Parameters
    ----------
    sig: numpy.ndarray
        The signal to analyze
    val: float
        The test value to find in the signal
    Returns
    -------
    int
    """
    return np.argmin(np.abs(sig-val))