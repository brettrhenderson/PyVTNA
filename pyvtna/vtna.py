import numpy as np
from sklearn.linear_model import LinearRegression
from vtna.signal import *
from scipy.optimize import Bounds, minimize
from vtna.align import *

class VTNA():

    def __init__(self, time, tonorm, normwith, handle_neg=replace_neg):
        # define a callback that can optionally be used to save the parameter value at each function call
        self.time = time.reshape(-1,1)
        self.tonorm = tonorm.reshape(-1,1)
        if normwith.ndim == 1:
            normwith = normwith.reshape(-1,1)
        self.normwith = normwith
        self.handle_neg = handle_neg
        self.handle_neg(self.normwith)
        self.bounds = Bounds(np.zeros(normwith.shape[1]), 4 * np.ones(normwith.shape[1]))
        self.orders = None
        self.k = None
        self.opt_history = []
        
    def smooth_data(self, win=5, win_type='blackman'):
        self.tonorm = smooth(self.tonorm.flatten(), window_len=win, window=win_type).reshape(-1,1)
        for col in self.normwith.T:
            col[:] = smooth(col, window_len=win, window=win_type)
        self.handle_neg(self.normwith)
        
    def order_fit(self, os):
        if (os < 0).any():
            return None, None
        # normalize time axis wrt normwith (a given rxn species)
        t_norm =  normalize_time(self.time, self.normwith, os)
        t_norm = t_norm.reshape(-1,1)
        # perform linear regression and calculate R^2
        return t_norm, LinearRegression().fit(t_norm, self.tonorm)

    # define an objective function that scores the fit of a given list of orders
    def order_score(self, os):
        t_norm, line = self.order_fit(os)
        if t_norm is not None:
            return -abs(line.score(t_norm, self.tonorm))
        else:
            return 0
    
    def set_bounds(self, lb, ub):
        if len(lb) != len(ub):
            print("lb and ub must have same length")
            return
        if len(lb) != self.normwith.shape[1]:
            print('lb and ub must have same number of entrys as number of reactants to find orders for')
        self.bounds = Bounds(lb, ub)
        
    def optimize(self, x0, method='nelder-mead', **kwargs):
        self.x0 = x0
        opt_res = minimize(self.order_score, x0, method=method, callback=lambda x: self.opt_history.append(x), **kwargs)
        self.orders = opt_res.x
        return opt_res.x
    
    def animate_opt(self):
        pass
        
    def get_orders(self):
        if self.orders is None:
            self.optimize()
        return self.orders
        
    def get_k(self):
        if self.k is None:
            if self.orders is None:
                self.optimize()
            line = self.order_fit(self.orders)[1]
            self.k = line.coef_[0][0]
        return self.k

def normalize_time(time, conc, order):
    dt = (time[1:] - time[:-1]).reshape(-1, 1)
    ave_conc = (conc[1:] + conc[:-1]) / 2
    # check if conc, order are iterables
    # if so, the integrand should have the product of the conc^order for each reagent
    if type(order) == np.ndarray and conc.shape[1] == len(order):
        integrand = dt
        for i, o in enumerate(order):
            integrand = integrand * ave_conc[:, i].reshape(-1,1)**o
    else:
        integrand = ((ave_conc.reshape(-1, 1))**order)*dt
    return np.concatenate((np.array([0]), np.cumsum(integrand, dtype=float)))


def order_search(t1, t2, prod1, prod2, reac1, reac2, o_range=(0,3), o_step=0.01, to_smooth=False, window_type='blackman',
                   win1=None, win2=None, smooth_mode='derivative', plot=False, truncate=False, overlap_thresh=0.25, **kwargs):
    """
    Returns the optimal order for a reactant in a rate law using the VTNA procedure.
    
    Given 2 reaction traces, each consisting of a reactant trace and 
    a product trace, where the reactant concentration differed between
    the two traces, compute the order of the reactant in the reaction
    rate law. This order, when used to normalize the time for each
    reaction trace by the reactant concentration, will cause the two
    reaction traces (product or reactant traces) to overlay onto each 
    other. This function maximizes the overlap between traces by 
    maximizing the pearson r correlation between the two normalized 
    reaction traces. To do this, the algorithm performs a grid search
    over a suitable range of reaction orders. The function assumes that
    reaction traces may have begun before the reaction actually started.
    Therefore, it attemts to remove "dead time" before the start of the
    reaction, where the traces are flatlined. It will optionally remove
    these dead-time data points or shift the reactions so that their actual 
    start times are at time=0.
    
    Parameters
    ----------
    t1 : numpy.ndarray
        An (N x 1) array representing time-series for reaction 1.
    t2 : numpy.ndarray
        An (N x 1) array representing time-series for reaction 2.    
    prod1 : numpy.ndarray
        An (N x 1) array representing time-series data. The first
        column contains the time of each measurement, and the 
        second column contains the value measured.
    prod2 : numpy.ndarray
        A second signal of the same form as `prod1`. Comes from 
        a second reaction with different reactant concentration.
    reac1 : numpy.ndarray
        The reactant signal from reaction 1, of the same form as `prod1`.
    reac2 : numpy.ndarray
        The reactant signal from reaction 2, of the same form as `prod1`.
    o_range : tuple, optional
        The range of order values to search, as (min, max). Default (0, 3).
    o_step : float, optional
        The step size for order grid search. Default 0.01.
    to_smooth : bool, optional
        If True, both signals will be smoothed before performing
        the grid search.
    window_type : str, optional
        Window type to be used for weighted window rolling mean smoothing
        {'flat', 'hanning', 'hamming', 'bartlett', 'blackman'}. Default='blackman'.
    win1 : int, optional
        The window size to be used for a rolling mean smoothing of 
        sig1.  If None, the algorithm will guess an optimal
        window size for the data
    win2 : int, optional
        The window size to be used for smoothing sig2.  Same 
        considerations apply as for `win1`.
    smooth_mode : str, optional
        {'derivative', 'standard', 'range'}
        Specifies the metric used to calculate the signal to noise
        ratio for finding the optimal window for smoothing. 
        In 'derivative' mode, SNR is the ratio of the maximum
        value of the absolute value of the derivative of the signal
        to the standard deviation of that derivative. In 'standard'
        mode, the SNR is calculated as the normal ratio of the signal
        mean to the signal standard deviation.  In 'range' mode, SNR
        is calculated as the ratio of the signal range to signal 
        standard deviation.
    plot : bool, optional
        If True, the final signal alignment will be plotted, with
        the signals both smoothed using an optimal smoothing
        algorithm and truncated so that the plot starts where the
        signal first spikes from baseline.
    truncate: bool, optional
        Whether to truncate the data to the best approximation of 
        where the reaction begins
    overlap_thresh: float, optional
        The proportion of a signal that must be overlapping during
        a grid search to consider the comparison valid. Default 0.25.
    kwargs : dict, optional
        Key word args to be passed in to matplotlib's pyplot.subplots
        function.  For example, figsize=(8, 6)
    
    Returns
    -------
    
    best_o : float
        The reactant order that maximizes trace overlap
    rs : list
        A record of all the Pearson correlation coefficients for each 
        of the orders tried, which are np.arange(o_range[0], o_range[1], o_step)
        
    """
    
    if win1 is None:
        win1 = get_best_win(prod1, mode=smooth_mode, minwin=1, maxwin=int(min(200, len(prod1) / 5)))
    if win2 is None:
        win2 = get_best_win(prod2, mode=smooth_mode, minwin=1, maxwin=int(min(200, len(prod2) / 5)))
        
    # 2. Shift signals such that their first points align
    if to_smooth:
        if truncate:
            # TODO: Fix truncate so it works with time_norm
            prod1, t1 = filter_shift(prod1, t1, window=win1)
            prod2, t2 = filter_shift(prod2, t2, window=win2)
        else:
            prod1 = smooth(prod1, window_len=win1, window=window_type)
            prod2 = smooth(prod2, window_len=win2, window=window_type)
            reac1 = smooth(reac1, window_len=win1, window=window_type)
            reac2 = smooth(reac2, window_len=win2, window=window_type)
    
    elif truncate:
        # TODO: Fix truncate so it works with time_norm
        s1, t1 = shift_zero(prod1, t1, window=win1)
        s2, t2 = shift_zero(prod2, t2, window=win2)
      
    best_o = 0
    best_tr = 0
    best_r = 0
    best_t1 = t1
    best_t2 = t2
    rs = []
    
    for o in np.arange(o_range[0], o_range[1], o_step):
        # a. normalize the time axis of both signals
        t1_norm = normalize_time(t1, reac1, o)
        t2_norm = normalize_time(t2, reac2, o)

        transtep = np.min(np.concatenate((t1_norm[1:] - t1_norm[:-1], t2_norm[1:] - t2_norm[:-1])))
        
        # b. determine suitable range for trans
        # TODO: figure something better out
        tol = 10*np.max(np.concatenate((t1_norm[1:] - t1_norm[:-1], t2_norm[1:] - t2_norm[:-1])))
        transrange = get_transrange(t1_norm, prod1, t2_norm, prod2, tolerance=tol)

        #for tran in np.arange(transrange[0], transrange[1], transtep):    
        # c. translate sig2
        t2_shift = t2_norm #+ tran

        # d. find the domain of overlap between sig1 and transformed sig2
        min_t = np.max([t1_norm[0], t2_shift[0]])
        max_t = np.min([t1_norm[-1], t2_shift[-1]])
        t1_over = t1_norm[(t1_norm >= min_t) & (t1_norm <= max_t)]
        t2_over = t2_shift[(t2_shift >= min_t) & (t2_shift <= max_t)]

        # e. check to make sure overlap is greater than some threshold
#         if len(t1_over) < overlap_thresh*len(t1_norm) or len(t2_over) < overlap_thresh*len(t2_norm):
#             continue

        # f. downsample the higher frequency signal over the overlap range
        #    to match the frequency of the lower frequency signal
        if len(t2_over) > len(t1_over):
            f = interp1d(t2_shift, prod2)
            prod2_int = f(t1_over)
            prod1_int = prod1[(t1_norm >= min_t) & (t1_norm <= max_t)]
        else:
            f = interp1d(t1_norm, prod1)
            prod1_int = f(t2_over)
            prod2_int = prod2[(t2_shift >= min_t) & (t2_shift <= max_t)]       

        # g. compute Pearson r coefficient 
        r = pear_r(prod1_int, prod2_int)
        rs.append(r)

        # h. check if r is the new best
        if abs(r) > best_r:
            best_r = r
            #best_tr = tran
            best_o = o
            best_t1 = t1_norm
            best_t2 = t2_shift 
    
    if plot:
        fig, (a1, a2) = plt.subplots(1, 2, **kwargs)
        a1.plot(t1, prod1, label="Rxn 1")
        a1.plot(t2, prod2, label='Rxn 2')
        a1.legend()
        a1.set_title('Original Product Traces for 2 Reactions')
        a2.plot(best_t1, prod1, label='Rxn 1')
        a2.plot(best_t2, prod2, label='Rxn 2')
        a2.legend()
        a2.set_title('Best-Fit Time-Normalization for Product Traces')
        print(f"Best fit achieved for an order of {best_o:0.2f}.")
        plt.show()
        plt.tight_layout()
    
    return best_o, rs


def VTNA_1D(time, tonorm, normwith, mino=0, maxo=3, res=0.01, sm=True, win=5, win_type='blackman', handle_neg=replace_neg):
    if sm:
        # smooth data
        normwith = smooth(normwith, window_len=5, window='blackman')
        tonorm = smooth(tonorm, window_len=5, window='blackman')

    tonorm = tonorm.reshape(-1,1)  # necessary for sklearn's linear regression

    # replace all negative values of the signals with 0, since these sometimes produce NaN during normalization
    handle_neg(normwith)

    rs = []
    bestr = 0
    besto = 0
    best_norm = None
    best_line = None

    for order in np.arange(mino, maxo, res):
        # normalize product (O2) time axis wrt N2O5 with guessed value for order
        t_norm =  normalize_time(time, normwith, order)
        t_norm = t_norm.reshape(-1,1)

        # perform linear regression and calculate R^2
        line = LinearRegression()  
        line.fit(t_norm, tonorm)
        r = line.score(t_norm, tonorm)
        rs.append(line.score(t_norm, tonorm))

        if abs(r) > abs(bestr):
            bestr = r
            best_norm = t_norm
            besto = order
            best_line = line
    return besto, best_line.coef_[0][0]