from scipy.optimize import Bounds, minimize
from scipy.interpolate import interp1d
from pyvtna.align import *
import pyvtna.metrics as metrics
from pyvtna.vtna import VTNA
from mpl_toolkits.axes_grid1 import make_axes_locatable

def poison_search(t1, t2, prod1, prod2, reac1, reac2, order, poison_range=None, nsteps=100, metric='RMSD',
                  smooth_traces=False, win1=None, win2=None, smooth_mode='derivative', smooth_cost_function=False,
                  win_cost=None, interp_cost_fun=False, window_type='blackman', plot=False, **kwargs):
    """
    Returns the optimal poisoning for a reactant with a known order in the rate law using a VTNA procedure.

    Given 2 reaction traces, each consisting of a reactant trace and
    a product trace, where the reactant concentration differed between
    the two traces, compute amount of reactant poisoning, where poisoning
    is a fixed amount removed from the reactant concentration from the start
    of the reaction. With an assumed order, finding the correct poisoning
    value will cause the two reaction traces (product or reactant traces)
    to overlay onto each other. This function maximizes the overlap between traces by
    maximizing the chosen overlap metric between the two normalized
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
    order : float, optional
        The assumed order of the reactant in the reaction rate law.
    poison_range : tuple, optional
        The range of poison concentrations to search
    nsteps : int, optional
        The number of different poison values in `poison_range` to try
    metric : str, optional
        Metric for calculating overlap of two signals.
        {'RMSD', 'R2', 'PearR', 'MAD'}. Default='RMSD'.
    smooth_traces : bool, optional
        If True, both signals will be smoothed before performing
        the grid search. Default: False
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
    smooth_cost_function : bool, Optional
        If True, the cost function estimated by the grid search will
        be smoothed before determining the optimal poisoning. Useful
        for noisy data, which causes a noisy cost function. Default: False
    win_cost : int, optional
        The size of the window used for smoothing the cost function. Must
        be smaller than the number of points in the grid_search. Default:
        half of the number of grid points.
    interp_cost_fun : bool, optional
        Whether to interpolate the cost function on a finer grid after smoothing.
        Can potentially allow a better answer with fewer grid points in the original search.
    window_type : str, optional
        Window type to be used for weighted window rolling mean smoothing
        {'flat', 'hanning', 'hamming', 'bartlett', 'blackman'}. Default='blackman'.
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
    f : `scipy.interpolate.interpolate.interp1d`
        The interpolated cost function. None if either `smooth_cost_function` or `interp_cost_fun` are False

    """
    if metric not in ['PearR', 'R2', 'RMSD', 'MAD']:
        raise ValueError(f'Chosen metric {metric} is not available. Try one of {{PearR, R2, RMSD, NSAD}}')
    metric_class = getattr(metrics, metric)
    metric = metric_class(max_is_best=True)

    if win1 is None:
        win1 = get_best_win(prod1, mode=smooth_mode, minwin=1, maxwin=int(min(200, len(prod1) / 5)))
    if win2 is None:
        win2 = get_best_win(prod2, mode=smooth_mode, minwin=1, maxwin=int(min(200, len(prod2) / 5)))

    prod1_orig = prod1
    prod2_orig = prod2
    if smooth_traces:
        prod1 = smooth(prod1, window_len=win1, window=window_type)
        prod2 = smooth(prod2, window_len=win2, window=window_type)
        reac1 = smooth(reac1, window_len=win1, window=window_type)
        reac2 = smooth(reac2, window_len=win2, window=window_type)

    if poison_range is None:
        poison_range = (0, min([min(reac1), min(reac2)]) / 2)
    poisonings = np.linspace(poison_range[0], poison_range[1], nsteps)
    overlaps = np.zeros(poisonings.shape)
    for i, pois in enumerate(poisonings):
        # a. normalize the time axis of both signals
        t1_norm = VTNA.normalize_time(t1, reac1 - pois, order)
        t2_norm = VTNA.normalize_time(t2, reac2 - pois, order)

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
        overlaps[i] = overlap

    original_overlaps = overlaps
    poisonings_cf = poisonings
    best_overlap, best_pois = None, None
    f = None
    if smooth_cost_function:
        if win_cost is None:
            win_cost = len(overlaps) // 2
        overlaps = smooth(overlaps, window_len=win_cost, window=window_type, general_sig=True)
        if interp_cost_fun:
            f = interp1d(poisonings, -overlaps, kind='quadratic')
            poisonings_cf = np.linspace(poison_range[0], poison_range[1], nsteps * 10)
            overlaps = -f(poisonings_cf)
            res = minimize(f, np.mean(poison_range), bounds=[poison_range], method='Nelder-Mead')
            best_overlap, best_pois = -res.fun, res.x[0]
    if best_overlap is None:
        best_overlap = overlaps.max()
        best_pois = poisonings[overlaps.argmax()]
    t1_norm = VTNA.normalize_time(t1, reac1, order)
    t2_norm = VTNA.normalize_time(t2, reac2, order)
    best_t1 = VTNA.normalize_time(t1, reac1 - best_pois, order)
    best_t2 = VTNA.normalize_time(t2, reac2 - best_pois, order)

    if plot:
        fig, (a1, a2, a3) = plt.subplots(1, 3, **kwargs)
        a1.scatter(t1_norm, prod1_orig, label="Rxn 1")
        a1.scatter(t2_norm, prod2_orig, label='Rxn 2')
        a1.legend()
        a1.set_title('Time-Normalized Product Traces')
        a2.plot(poisonings_cf, overlaps, c='tab:blue', linewidth=2, label="Cost Function")
        if smooth_cost_function:
            a2.plot(poisonings, original_overlaps, c='tab:orange', linewidth=2, label="Un-Smoothed Cost Function")
        a2.scatter([best_pois], best_overlap, c="tab:red", s=100, label='Best Overlap')
        a2.legend()
        a2.set_title('Overlap Metric For Scanned Poisonings')
        a3.scatter(best_t1, prod1_orig, label='Rxn 1')
        a3.scatter(best_t2, prod2_orig, label='Rxn 2')
        a3.legend()
        a3.set_title(f'Best-Fit Poisoning for Product Traces (Order={order})')
        print(f"Best fit achieved for an poisoning of {best_pois:0.2f}.")
        plt.tight_layout()
        plt.show()

    return best_pois, overlaps, f


def poison_opt(t1, t2, prod1, prod2, reac1, reac2, order=None, poison_range=None, method='Nelder-Mead', metric='RMSD',
               to_smooth=False, window_type='blackman', win1=None, win2=None, smooth_mode='derivative',
               plot=False, **kwargs):
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
    order : float, optional
        The assumed order of the reactant in the reaction rate law.
    poison_range : tuple, optional
        The range of poison concentrations to search.
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

    if win1 is None:
        win1 = get_best_win(prod1, mode=smooth_mode, minwin=1, maxwin=int(min(200, len(prod1) / 5)))
    if win2 is None:
        win2 = get_best_win(prod2, mode=smooth_mode, minwin=1, maxwin=int(min(200, len(prod2) / 5)))

    prod1_orig = prod1
    prod2_orig = prod2
    if to_smooth:
        prod1 = smooth(prod1, window_len=win1, window=window_type)
        prod2 = smooth(prod2, window_len=win2, window=window_type)
        if (hasattr(reac1, '__iter__') and hasattr(reac2, '__iter__')):
            reac1 = smooth(reac1, window_len=win1, window=window_type)
            reac2 = smooth(reac2, window_len=win2, window=window_type)

    ps = []
    overlaps = []

    # The ojective function to minimize
    def deviation(p):
        p = p[0]
        # a. normalize the time axis of both signals
        t1_norm = VTNA.normalize_time(t1, reac1 - p, order)
        t2_norm = VTNA.normalize_time(t2, reac2 - p, order)

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
        ps.append(p)
        overlaps.append(overlap)
        return -overlap

    # The minimization algorithm
    if poison_range is None:
        poison_range = (0, min([min(reac1), min(reac2)]) / 2)
    start_p = np.mean(poison_range)

    result = minimize(deviation, start_p, method=method, bounds=[poison_range])

    if plot:
        fig, (a1, a2, a3) = plt.subplots(1, 3, **kwargs)
        a1.scatter(VTNA.normalize_time(t1, reac1, order), prod1_orig, label="Rxn 1")
        a1.scatter(VTNA.normalize_time(t2, reac2, order), prod2_orig, label='Rxn 2')
        a1.legend()
        a1.set_title('Original Product Traces for 2 Reactions')
        s2 = a2.scatter(ps, overlaps, c=range(len(ps)), linewidth=2, label="Iterations")
        divider = make_axes_locatable(a2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(s2, cax=cax)
        # a2.scatter(result.x, [-result.fun], c="tab:red", s=100, label='Best Overlap')
        a2.legend()
        a2.set_title('Overlap Metric For Scanned Orders')
        a3.scatter(VTNA.normalize_time(t1, reac1 - result.x[0], order), prod1_orig, label='Rxn 1')
        a3.scatter(VTNA.normalize_time(t2, reac2 - result.x[0], order), prod2_orig, label='Rxn 2')
        a3.legend()
        a3.set_title('Best-Fit Time-Normalization for Product Traces')
        print(f"Best fit achieved for an order of {result.x[0]:0.2f}.")
        plt.tight_layout()
        plt.show()

    return result, {'poisonings': ps, 'overlaps': overlaps}