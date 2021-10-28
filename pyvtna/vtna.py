from sklearn.linear_model import LinearRegression
from scipy.optimize import Bounds, minimize
from pyvtna.align import *
import pyvtna.metrics as metrics
from pyvtna.signal import is_ascending
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


def order_search(t1, t2, prod1, prod2, reac1, reac2, o_range=(0,3), o_step=0.01,  metric='RMSD', to_smooth=False, window_type='blackman',
                   win1=None, win2=None, smooth_mode='derivative', plot=False, **kwargs):
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
    over a suitable range of reaction orders.
    
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
    metric : str, optional
        Metric for calculating overlap of two signals.
        {'RMSD', 'R2', 'PearR', 'MAD'}. Default='RMSD'.
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
    if metric not in ['PearR', 'R2', 'RMSD', 'MAD']:
        raise ValueError(f'Chosen metric {metric} is not available. Try one of {{PearR, R2, RMSD, NSAD}}')
    metric_class = getattr(metrics, metric)
    metric = metric_class(max_is_best=True)

    # configure so that the largest overlap is always the highest value
    if not is_ascending(prod1):
        metric = lambda x, y: -metric(x, y)
    
    if win1 is None:
        win1 = get_best_win(prod1, mode=smooth_mode, minwin=1, maxwin=int(min(200, len(prod1) / 5)))
    if win2 is None:
        win2 = get_best_win(prod2, mode=smooth_mode, minwin=1, maxwin=int(min(200, len(prod2) / 5)))

    prod1_orig = prod1
    prod2_orig = prod2
    if to_smooth:
        prod1 = smooth(prod1, window_len=win1, window=window_type)
        prod2 = smooth(prod2, window_len=win2, window=window_type)
        reac1 = smooth(reac1, window_len=win1, window=window_type)
        reac2 = smooth(reac2, window_len=win2, window=window_type)
      
    best_order = 0
    best_overlap = -1000
    best_t1 = t1
    best_t2 = t2
    overlaps = []

    os = np.arange(o_range[0], o_range[1], o_step)
    for o in os:
        # a. normalize the time axis of both signals
        t1_norm = normalize_time(t1, reac1, o)
        t2_norm = normalize_time(t2, reac2, o)

        # d. find the domain of overlap between sig1 and transformed sig2
        min_t = np.max([t1_norm[0], t2_norm[0]])
        max_t = np.min([t1_norm[-1], t2_norm[-1]])
        t1_over = t1_norm[(t1_norm >= min_t) & (t1_norm <= max_t)]
        t2_over = t2_norm[(t2_norm >= min_t) & (t2_norm <= max_t)]

        # f. downsample the higher frequency signal over the overlap range
        #    to match the frequency of the lower frequency signal
        if len(t2_over) > len(t1_over):
            f = interp1d(t2_norm, prod2)
            prod2_int = f(t1_over)
            prod1_int = prod1[(t1_norm >= min_t) & (t1_norm <= max_t)]
        else:
            f = interp1d(t1_norm, prod1)
            prod1_int = f(t2_over)
            prod2_int = prod2[(t2_norm >= min_t) & (t2_norm <= max_t)]

        # g. compute Pearson r coefficient 
        overlap = metric(prod1_int, prod2_int)
        overlaps.append(overlap)

        # h. check if r is the new best
        if overlap > best_overlap:
            best_overlap = overlap
            best_order = o
            best_t1 = t1_norm
            best_t2 = t2_norm
    
    if plot:
        fig, (a1, a2, a3) = plt.subplots(1, 3, **kwargs)
        a1.scatter(t1, prod1_orig, label="Rxn 1")
        a1.scatter(t2, prod2_orig, label='Rxn 2')
        a1.legend()
        a1.set_title('Original Product Traces for 2 Reactions')
        a2.plot(os, overlaps, c='tab:blue', linewidth=2, label="Overlap Scan")
        a2.scatter([best_order], best_overlap, c="tab:red", s=100, label='Best Overlap')
        a2.legend()
        a2.set_title('Overlap Metric For Scanned Orders')
        a3.scatter(best_t1, prod1_orig, label='Rxn 1')
        a3.scatter(best_t2, prod2_orig, label='Rxn 2')
        a3.legend()
        a3.set_title('Best-Fit Time-Normalization for Product Traces')
        print(f"Best fit achieved for an order of {best_order:0.2f}.")
        plt.tight_layout()
        plt.show()
    
    return best_order, overlaps


def order_opt(t1, t2, prod1, prod2, reac1, reac2, o_range=None, method='Nelder-Mead', metric='RMSD', to_smooth=False,
                 window_type='blackman', win1=None, win2=None, smooth_mode='derivative', plot=False, **kwargs):
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
    reaction traces.

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
        The range of order values to search, as (min, max). Default None.
        These bounds will be ignored unless `method` is one of the follwing:
        "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", or "trust-constr-...".
    method : str, optional
        The minimization algorithm to use. Can take any of the values allowed
        by `scipy.optimize.minimize`. Default "Nelder-Mead".
    metric : str, optional
        Metric for calculating overlap of two signals.
        {'RMSD', 'R2', 'PearR', 'MAD'}. Default='RMSD'.
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
    if metric not in ['PearR', 'R2', 'RMSD', 'MAD']:
        raise ValueError(f'Chosen metric {metric} is not available. Try one of {{PearR, R2, RMSD, NSAD}}')
    metric_class = getattr(metrics, metric)
    metric = metric_class(max_is_best=True)

    # configure so that the largest overlap is always the highest value
    if not is_ascending(prod1):
        metric = lambda x, y: -metric(x, y)

    if win1 is None:
        win1 = get_best_win(prod1, mode=smooth_mode, minwin=1, maxwin=int(min(200, len(prod1) / 5)))
    if win2 is None:
        win2 = get_best_win(prod2, mode=smooth_mode, minwin=1, maxwin=int(min(200, len(prod2) / 5)))

    prod1_orig = prod1
    prod2_orig = prod2
    if to_smooth:
        prod1 = smooth(prod1, window_len=win1, window=window_type)
        prod2 = smooth(prod2, window_len=win2, window=window_type)
        reac1 = smooth(reac1, window_len=win1, window=window_type)
        reac2 = smooth(reac2, window_len=win2, window=window_type)

    os = []
    overlaps = []

    # The ojective function to minimize
    def deviation(o):
        o = o[0]
        # a. normalize the time axis of both signals
        t1_norm = normalize_time(t1, reac1, o)
        t2_norm = normalize_time(t2, reac2, o)

        # d. find the domain of overlap between sig1 and transformed sig2
        min_t = np.max([t1_norm[0], t2_norm[0]])
        max_t = np.min([t1_norm[-1], t2_norm[-1]])
        t1_over = t1_norm[(t1_norm >= min_t) & (t1_norm <= max_t)]
        t2_over = t2_norm[(t2_norm >= min_t) & (t2_norm <= max_t)]

        # f. downsample the higher frequency signal over the overlap range
        #    to match the frequency of the lower frequency signal
        if len(t2_over) > len(t1_over):
            f = interp1d(t2_norm, prod2)
            prod2_int = f(t1_over)
            prod1_int = prod1[(t1_norm >= min_t) & (t1_norm <= max_t)]
        else:
            f = interp1d(t1_norm, prod1)
            prod1_int = f(t2_over)
            prod2_int = prod2[(t2_norm >= min_t) & (t2_norm <= max_t)]

        # g. overlap
        overlap = metric(prod1_int, prod2_int)
        os.append(o)
        overlaps.append(overlap)
        return -overlap

    # The minimization algorithm
    start_o = np.array([0])
    if o_range is not None:
        start_o = np.array([(o_range[0] + o_range[1]) / 2])
        o_range = [o_range]

    result = minimize(deviation, start_o, method=method, bounds=o_range)

    if plot:
        fig, (a1, a2, a3) = plt.subplots(1, 3, **kwargs)
        a1.scatter(t1, prod1_orig, label="Rxn 1")
        a1.scatter(t2, prod2_orig, label='Rxn 2')
        a1.legend()
        a1.set_title('Original Product Traces for 2 Reactions')
        s2 = a2.scatter(os, overlaps, c=range(len(os)), linewidth=2, label="Iterations")
        divider = make_axes_locatable(a2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(s2, cax=cax)
        # a2.scatter(result.x, [-result.fun], c="tab:red", s=100, label='Best Overlap')
        a2.legend()
        a2.set_title('Overlap Metric For Scanned Orders')
        a3.scatter(normalize_time(t1, reac1, result.x[0]), prod1_orig, label='Rxn 1')
        a3.scatter(normalize_time(t2, reac2, result.x[0]), prod2_orig, label='Rxn 2')
        a3.legend()
        a3.set_title('Best-Fit Time-Normalization for Product Traces')
        print(f"Best fit achieved for an order of {result.x[0]:0.2f}.")
        plt.tight_layout()
        plt.show()

    return result, {'orders': os, 'deviations': overlaps}

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
