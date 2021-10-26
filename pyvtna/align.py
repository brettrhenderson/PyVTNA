import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pyvtna.signal import *
from pyvtna.metrics import *

def align_starts(sig1, sig2, time1=None, time2=None):
    """
    Align the starting times of two signals, shifting them to the same zero point.
    
    The respective time arrays for the signals will be aligned if provided, else the signals
    will be aligned by index (assuming the same time array for each).
    
    Parameters
    ----------
    sig1: numpy.ndarray
        The first signal to align
    sig2: numpy.ndarray
        The second signal to align
    time1: numpy.ndarray, optional
        The time array for signal 1
    time2: numpy.ndarray, optional
        The time array for signal 2
    """
    if time1 is None:
        if is_ascending(sig1) and is_ascending(sig2):
            # which one starts higher
            if sig1[0] > sig2[0]:
                return sig1, sig2[closest_idx(sig2, sig1[0]):]
            else:
                return sig1[closest_idx(sig1, sig2[0]):], sig2
        elif not is_ascending(sig1) and not is_ascending(sig2):
            if sig1[0] < sig2[0]:
                return sig1, sig2[closest_idx(sig2, sig1[0]):]
            else:
                return sig1[closest_idx(sig1, sig2[0]):], sig2
        else:
            print("Both signals need to be either ascending or descending.")
    elif time2 is not None:
        if is_ascending(sig1) and is_ascending(sig2):
            # which one starts higher
            if sig1[0] > sig2[0]:
                return (sig1, sig2[closest_idx(sig2, sig1[0]):], 
                        time1, time2[closest_idx(sig2, sig1[0]):] - time2[closest_idx(sig2, sig1[0])])
            else:
                return (sig1[closest_idx(sig1, sig2[0]):], sig2, 
                        time1[closest_idx(sig1, sig2[0]):] - time1[closest_idx(sig1, sig2[0])], time2)
        elif not is_ascending(sig1) and not is_ascending(sig2):
            if sig1[0] < sig2[0]:
                return (sig1, sig2[closest_idx(sig2, sig1[0]):], 
                        time1, time2[closest_idx(sig2, sig1[0]):] - time2[closest_idx(sig2, sig1[0])])
            else:
                return (sig1[closest_idx(sig1, sig2[0]):], sig2, 
                        time1[closest_idx(sig1, sig2[0]):] - time1[closest_idx(sig1, sig2[0])], time2)
        else:
            print("Both signals need to be either ascending or descending.")
    else:
        raise ValueError("time1 and time2 must either be both None or both arrays")

def grid_search_comp(sig1, sig2, smooth=False, win1=None, win2=None, smooth_mode='derivative', plot=False,
                     truncate=False, overlap_thresh=0.25, comp_res=0.01, comp_tol=0.1, **kwargs):
    """
    Returns the compression factor used to make sig2 match sig1.
    
    Given 2 signals, sig1 and sig2, where the signals may have both
    been translated and dilated relative to one another, this
    function computes the factor, which when used to scale the time
    axis of sig2, maximizes the pearson r correlation between
    sig1 and sig2.  To do this, the algorithm performs a grid search
    for the combination of translation and compression of signal 2 
    that maximizes the Pearson r coefficient for signal 1 and this
    transformed signal 2.
    
    Parameters
    ----------
    
    sig1 : numpy.ndarray
           An (N x 2) array representing time-series data. The first
           column contains the time of each measurement, and the 
           second column contains the value measured.
    sig2 : numpy.ndarray
           A second signal of the same form as `sig1`.
    
    smooth : bool, optional
             If True, both signals will be smoothed before performing
             the grid search.
    
    win1 : int, optional
           The window size to be used for a rolling mean smoothing of 
           sig1.  If None, the algorithm will guess an optimal
           window size for the data
           
    win2 : int, optional
           The window size to be used for smoothing sig2.  Same 
           considerations apply as for `win1`.
           
    plot : bool, optional
           If True, the final signal alignment will be plotted, with
           the signals both smoothed using an optimal smoothing
           algorithm and truncated so that the plot starts where the
           signal first spikes from baseline.
           
    overlap_thresh: float, optional
             The proportion of a signal that must be overlapping during
             a grid search to consider the comparison valid. Default 0.25.
             
    truncate: bool, optional
              Whether to truncate the data to the best approximation of 
              where the reaction begins
    
    kwargs : optional
             Key word args to be passed in to matplotlib's pyplot.subplots
             function.  For example, figsize=(8, 6)
    
    Returns
    -------
    
    comp : float
           The optimal compression factor for sig2
    
    """
    
     # 0. Prepare signal inputs
    t1, s1 = sig1[:, 0], sig1[:, 1]
    t2, s2 = sig2[:, 0], sig2[:, 1]
    
    if win1 is None:
        win1 = get_best_win(s1, mode=smooth_mode, minwin=1, maxwin=int(min(200, len(s1) / 5)))
    if win2 is None:
        win2 = get_best_win(s2, mode=smooth_mode, minwin=1, maxwin=int(min(200, len(s2) / 5)))
        
    # 2. Shift signals such that their first points align
    if smooth:
        if truncate:
            s1, t1 = filter_shift(s1, t1, win=win1)
            s2, t2 = filter_shift(s2, t2, win=win2)
        else:
            s1, t1 = rolling_mean(s1, win=win1), x_rolling(t1, win=win1)
            s2, t2 = rolling_mean(s2, win=win2), x_rolling(t2, win=win2)        
    elif truncate:
        s1, t1 = shift_zero(s1, t1, window=win1)
        s2, t2 = shift_zero(s2, t2, window=win2)
      
    
    def grid_search(comprange, compstep, transtep):
        best_c = 0
        best_tr = 0
        best_r = 0
        best_sig = np.zeros(len(t1))
        best_t = t1
        
        for comp in np.arange(comprange[0], comprange[1], compstep):
            # a. compress sig2
            t2_comp = t2 * comp
            
            # b. determine suitable range for trans
            tol = 10*np.max(np.concatenate((t1[1:] - t1[:-1], t2[1:] - t2[:-1])))
            transrange = get_transrange(t1, s1, t2_comp, s2, tolerance=tol)
            
            for tran in np.arange(transrange[0], transrange[1], transtep):    
                # c. translate sig2
                t2_shift = t2_comp + tran
                
                # d. find the domain of overlap between sig1 and transformed sig2
                min_t = np.max([np.min(t1), np.min(t2_shift)])
                max_t = np.min([np.max(t1), np.max(t2_shift)])
                t1_over = t1[(t1 >= min_t) & (t1 <= max_t)]
                t2_over = t2_shift[(t2_shift >= min_t) & (t2_shift <= max_t)]
                
                # e. check to make sure overlap is greater than some threshold
                if len(t1_over) < overlap_thresh*len(t1) or len(t2_over) < overlap_thresh*len(t2):
                    continue
                
                # f. downsample the higher frequency signal over the overlap range
                #    to match the frequency of the lower frequency signal
                if len(t2_over) > len(t1_over):
                    f = interp1d(t2_shift, s2)
                    sig2_int = f(t1_over)
                    sig1_int = s1[(t1 >= min_t) & (t1 <= max_t)]
                else:
                    f = interp1d(t1, s1)
                    sig1_int = f(t2_over)
                    sig2_int = s2[(t2_shift >= min_t) & (t2_shift <= max_t)]       
                
                # g. compute Pearson r coefficient 
                r = pear_r(sig1_int, sig2_int)
                
                # h. check if r is the new best
                if abs(r) > best_r:
                    best_r = r
                    best_tr = tran
                    best_c = comp
                    best_sig = s2
                    best_t = t2_shift   
        return best_t, best_sig, best_tr, best_c 
                
                
    # 2. Coarse-grained grid search
    comp_ratio = get_interpercentile_diff(s1, t1, 25, 75) / get_interpercentile_diff(s2, t2, 25, 75)
    comp_range = (comp_ratio - comp_tol * comp_ratio, comp_ratio + comp_tol * comp_ratio)
    tran_step_base = np.min(np.concatenate((t1[1:] - t1[:-1], t2[1:] - t2[:-1])))
    best_t, best_sig, trans, comp = grid_search(comp_range, comp_res, tran_step_base)
    
    # 3. Fine-grained grid search
#     best_t, best_sig, trans, comp = grid_search((comp - comp_res*10, comp + comp_res*10), comp_res, tran_step_base)
    
    if plot:
        fig, a1 = plt.subplots(1, 1, **kwargs)
        a1.plot(t1, s1, label="Sig 1")
        a1.plot(t2, s2, label='Original Sig 2', color='k', alpha=0.6)
        a1.plot(best_t, best_sig, label='Transformed Sig 2')
        a1.legend()
        a1.set_title('Compression and Translation of Signal 2 For Best Fit')
        print(f"Best fit achieved for a compression of {comp:0.2f} and subsequent translation of {trans:0.2f} applied to signal 2.")
        plt.show()
        plt.tight_layout()
    
    return trans, comp


def get_transrange(t1, s1, t2, s2, tolerance=0.01, p=25):
    """
    Get the difference in times between the same percentile in two signals.
    
    Includes an extra specified tolerance.
    
    Parameters
    ----------
    t1: numpy.ndarray
        The time sequence for the first signal.
    s1: numpy.ndarray
        The first signal.
    t2: numpy.ndarray
        The time sequence for the second signal.
    s2: numpy.ndarray
        The second signal.
    tolerance: float, optional
        An extra value to be added on to each side of the time window for extra cushioning. Default 0.01.
    p: float
        The percentile to evaluate the time difference for.
    Returns
    -------
    (float, float)
    
    """
    i1 = get_percentile_idx(s1, p)
    i2 = get_percentile_idx(s2, p)
    ti1 = t1[i1]
    #tol = tolerance * np.max(t1) - np.min(t1)  # t1[-1] - t1[0] if time is monotonic  # relative tolerance
    tol = tolerance # tolerance is an absolute value
    ti2 = t2[i2]
    return (ti1 - tol - ti2, ti1 + tol - ti2)

def get_percentile_idx(sig, percentile, mode='derivative', minwin=1, maxwin=None):
    """
    Get the index of the value in the signal that is at a certain percentile of the max signal value.
    
    Signals must be normalized to values between 0 and 1, and the signal is smoothed before evaluating.
    
    Parameters
    ----------
    sig: numpy.ndarray
        The signal to analyze.
    percentile: float
        The percentile to find the index of. Should be between 0 and 100
    mode: str, optional {'derivative', 'standard', 'range'}
        The mode used to get the best window for smoothing. Default 'derivative'.
    minwin: int, optional
        The minimum windowsize for smoothing. Default 1.
    maxwin: int, optional
        The maximum windowsize for smoothing.
    Returns
    -------
    int
    """
    if maxwin is None:
        maxwin = len(sig) // 5
    # try 1.  Smooth signal, then select the first point greater/less than percentile
    win = get_best_win(sig, minwin=minwin, maxwin=maxwin, mode=mode)
    smooth = rolling_mean(sig, win=win)
    ran = np.max(sig) - np.min(sig)
    
    if is_ascending(sig):
        for i, el in enumerate(sig):
            if el > percentile / 100 * ran + np.min(sig):
                return i
    else:
        for i, el in enumerate(sig):
            if 1 - el > percentile / 100 * ran + np.min(sig):
                return i
            
def get_interpercentile_diff(sig, time, p1, p2, **kwargs):
    """
    Get the the time between the signal reaching percentile p1 and percentile p2.
    
    Signals must be normalized to values between 0 and 1, and the signal is smoothed before evaluating.
    
    Parameters
    ----------
    sig: numpy.ndarray
        The signal to analyze.
    time: numpy.ndarray
        The time sequence of the signal
    p1: float
        The first percentile to start the window with.
    p2: float
        The second percentile to end the window with.
    **kwargs:
        Other keyword arguments to pass to `get_percentile_idx`.
    Returns
    -------
    float
    """
    i1 = get_percentile_idx(sig, p1, **kwargs)
    i2 = get_percentile_idx(sig, p2, **kwargs)
    return time[i2] - time[i1]
