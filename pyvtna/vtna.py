from sklearn.linear_model import LinearRegression
from scipy.optimize import Bounds, minimize
from scipy.interpolate import interp1d, interp2d
from pyvtna.align import *
import itertools
import pyvtna.metrics as metrics
import copy


class VTNA:

    def __init__(self, reader, writer=None, visualizer=None, overlap_metric='RMSD'):
        self.data = reader
        self.writer = writer
        self.visualizer = visualizer
        self.poisonings = [0 for spec in self.data.species_names]
        self.k = 'k'
        self.orders = [0 for spec in self.data.species_names]
        if overlap_metric not in ['PearR', 'R2', 'RMSD', 'MAD']:
            raise ValueError(f'Chosen metric {overlap_metric} is not available. Try one of {{PearR, R2, RMSD, NSAD}}')
        metric_class = getattr(metrics, overlap_metric)
        self.overlap_metric = metric_class(max_is_best=True)

    def load(self, data_loc):
        self.data.load(data_loc)

    def reset(self):
        self.data.reset_reaction_traces()
        self.k = 'k'
        self.orders = [0 for spec in self.data.species_names]
        
    def set_overlap_metric(self, overlap_metric):
        if overlap_metric not in ['PearR', 'R2', 'RMSD', 'MAD']:
            raise ValueError(f'Chosen metric {overlap_metric} is not available. Try one of {{PearR, R2, RMSD, NSAD}}')
        metric_class = getattr(metrics, overlap_metric)
        self.overlap_metric = metric_class(max_is_best=True)

    def add_catalysts(self, catalyst_concentrations, catalyst_name='catalyst'):
        for rxn_name, cat_conc in catalyst_concentrations.items():
            rxn_trace = self.data.reaction_traces[rxn_name]
            augmented_trace = np.c_[rxn_trace, np.ones(rxn_trace.shape[0]) * cat_conc]
            self.data.reaction_traces[rxn_name] = augmented_trace
        self.data.species_names.append(catalyst_name)
        self.orders.append(0)

    def update_reaction_name(self, old_name, new_name):
        self.data.reaction_traces[new_name] = self.data.reaction_traces.pop(old_name)
        idx = self.data.reaction_names.index(old_name)
        self.data.reaction_names[idx] = new_name

    def update_species_name(self, old_name, new_name):
        idx = self.data.species_names.index(old_name)
        self.data.species_names[idx] = new_name

    def get_current_order(self, species_name):
        idx = self.data.species_names.index(species_name)
        return self.orders[idx]

    def get_current_poisoning(self, species_name):
        idx = self.data.species_names.index(species_name)
        return self.poisonings[idx]

    def set_rate_constant(self, k):
        self.k = k

    def pprint_rate_law(self):
        if isinstance(self.k, str):
            rate_law = f'Rate = {self.k}'
        else:
            rate_law = f'Rate = {self.k: 0.2e}'
        for spec_name in self.data.species_names:
            order = self.get_current_order(spec_name)
            if order:
                rate_law += f'[{spec_name}]^{order}'
        print(rate_law)

    def set_reactant_order(self, spec_to_norm_by, order):
        new_order = order
        order = order - self.get_current_order(spec_to_norm_by)
        if order:
            for rxn_name, rxn_trace in self.data.reaction_traces.items():
                time = rxn_trace[:, 0]
                conc = rxn_trace[:, self.data.species_names.index(spec_to_norm_by) + 1]
                t_norm = self.normalize_time(time, conc, order)
                rxn_trace[:, 0] = t_norm
        self.orders[self.data.species_names.index(spec_to_norm_by)] = new_order

    def set_species_poisoning(self, species_name, poisoning):
        relative_poisoning = poisoning - self.get_current_poisoning(species_name)
        spec_idx = self.data.species_names.index(species_name)
        for rxn_name, rxn_trace in self.data.reaction_traces.items():
            rxn_trace[:, spec_idx + 1] -= relative_poisoning
        self.poisonings[spec_idx] = poisoning

    def smooth_data(self, smooth_mode='derivative', win_type='blackman'):
        for rxn_name in self.data.reaction_names:
            self.smooth_data_by_reaction(rxn_name, smooth_mode=smooth_mode, win_type=win_type)

    def smooth_data_by_reaction(self, rxn_name, win=None, smooth_mode='derivative', win_type='blackman'):
        rxn_trace = self.data.reaction_traces[rxn_name]
        for spec in rxn_trace.T[1:]:
            if win is None:
                win = get_best_win(spec, mode=smooth_mode, minwin=1, maxwin=int(min(200, len(spec) // 5)))
            spec[:] = smooth(spec, window_len=win, window=win_type)

    def smooth_data_by_reaction_and_species(self, rxn_name, spec_name, win=None, smooth_mode='derivative',
                                            win_type='blackman'):
        rxn_trace = self.data.reaction_traces[rxn_name]
        if not isinstance(spec_name, int):
            if isinstance(spec_name, str):
                spec_name = self.data.species_names.index(spec_name)
            else:
                raise ValueError("Species must be either a list of integer indexes or strings identifying species names")
        spec = rxn_trace.T[spec_name]
        if win is None:
            win = get_best_win(spec, mode=smooth_mode, minwin=1, maxwin=int(min(200, len(spec) // 5)))
        spec[:] = smooth(spec, window_len=win, window=win_type)

    def get_average_slope(self, reaction_name, species_name):
        t = self.data.reaction_traces[reaction_name][:, 0]
        spec_trace = self.data.reaction_traces[reaction_name][:, self.data.species_names.index(species_name) + 1]
        t = t.reshape(-1, 1)
        # perform linear regression and calculate R^2
        line = LinearRegression().fit(t, spec_trace)
        return line.coef_[0]

    @staticmethod
    def calculate_overlap(t1, t2, trace1, trace2, metric):
        # find the domain of overlap between trace1 and trace2
        min_t = np.max([t1[0], t2[0]])
        max_t = np.min([t1[-1], t2[-1]])
        t1_over = t1[(t1 >= min_t) & (t1 <= max_t)]
        t2_over = t2[(t2 >= min_t) & (t2 <= max_t)]

        # downsample the higher frequency signal over the overlap range
        #    to match the frequency of the lower frequency signal
        if len(t2_over) > len(t1_over):
            f = interp1d(t2, trace2)
            trace2_int = f(t1_over)
            trace1_int = trace1[(t1 >= min_t) & (t1 <= max_t)]
        else:
            f = interp1d(t1, trace1)
            trace1_int = f(t2_over)
            trace2_int = trace2[(t2 >= min_t) & (t2 <= max_t)]

        # overlap
        overlap = metric(trace1_int, trace2_int)
        return overlap

    @staticmethod
    def calculate_linearity(time, to_norm, to_norm_by, order):
        # normalize time axis wrt normwith (a given rxn species)
        t_norm = VTNA.normalize_time(time, to_norm_by, order)
        t_norm = t_norm.reshape(-1, 1)
        # perform linear regression and calculate R^2
        line = LinearRegression().fit(t_norm, to_norm)
        return -abs(line.score(t_norm, to_norm)), line

    @staticmethod
    def normalize_time(time, conc, order):
        dt = (time[1:] - time[:-1]).reshape(-1, 1)

        # check if concentrations are single values, indicating catalyst or excess reagent
        if isinstance(conc, float):
            conc = conc * np.ones((len(time), 1))
        elif len(conc) == 1:
            conc = np.array(conc)
            conc = conc * np.ones((len(time), conc.shape[1]))

        ave_conc = (conc[1:] + conc[:-1]) / 2
        # check if conc, order are iterables
        # if so, the integrand should have the product of the conc^order for each reagent
        if type(order) == np.ndarray and conc.shape[1] == len(order):
            integrand = dt
            for i, o in enumerate(order):
                integrand = integrand * ave_conc[:, i].reshape(-1, 1) ** o
        else:
            integrand = ((ave_conc.reshape(-1, 1)) ** order) * dt
        return np.concatenate((np.array([0]), np.cumsum(integrand, dtype=float)))

    def order_search(self, rxn_1, rxn_2, spec_to_norm, spec_to_norm_by, o_range=(0, 3), nsteps=100,
                     smooth_cost_function=True, win_cost=None, interp_cost_fun=True, window_type='blackman'):

        """
        Returns the optimal order for a reactant in a rate law using the VTNA procedure.

        Given 2 reaction traces, each consisting of a reactant trace and
        a product trace, where the reactant concentration differed between
        the two traces, compute the order of the reactant in the reaction
        rate law. This order, when used to normalize the time for each
        reaction trace by the reactant concentration, will cause the two
        reaction traces (product or reactant traces) to overlay onto each
        other. This function maximizes the overlap between traces by
        maximizing the chosen overlap metric between the two normalized
        reaction traces. To do this, the algorithm performs a grid search
        over a suitable range of reaction orders.

        Parameters
        ----------
        rxn_1 : str
            Name of the first reaction.
        rxn_2 : str
            Name of the second reaction.
        spec_to_norm : str
            The name of the species whose trace should be normalized.
        spec_to_norm_by : str
            The name of the species whose trace should be integrated in order to normalize
            the time axis.
        o_range : tuple, optional
            The range of order values to search, as (min, max). Default (0, 3).
        nsteps : float, optional
            The number of steps in `o_range` for order grid search. Default 100.
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

        t1 = self.data.reaction_traces[rxn_1][:, 0]
        t2 = self.data.reaction_traces[rxn_2][:, 0]
        to_norm1 = self.data.reaction_traces[rxn_1][:, self.data.species_names.index(spec_to_norm) + 1]
        to_norm2 = self.data.reaction_traces[rxn_2][:, self.data.species_names.index(spec_to_norm) + 1]
        to_norm_by1 = self.data.reaction_traces[rxn_1][:, self.data.species_names.index(spec_to_norm_by) + 1]
        to_norm_by2 = self.data.reaction_traces[rxn_2][:, self.data.species_names.index(spec_to_norm_by) + 1]

        orders = np.linspace(o_range[0], o_range[1], nsteps)
        overlaps = np.zeros(orders.shape)
        for i, o in enumerate(orders):
            # a. normalize the time axis of both signals
            t1_norm = self.normalize_time(t1, to_norm_by1, o)
            t2_norm = self.normalize_time(t2, to_norm_by2, o)
            overlaps[i] = self.calculate_overlap(t1_norm, t2_norm, to_norm1, to_norm2, self.overlap_metric)

        original_overlaps = overlaps
        orders_cf = orders
        best_overlap, best_order = None, None
        f = None
        if smooth_cost_function:
            if win_cost is None:
                win_cost = len(overlaps) // 2
            overlaps = smooth(overlaps, window_len=win_cost, window=window_type, general_sig=True)
            if interp_cost_fun:
                f = interp1d(orders, -overlaps, kind='quadratic')
                orders_cf = np.linspace(o_range[0], o_range[1], nsteps * 10)
                overlaps = -f(orders_cf)
                res = minimize(f, np.mean(o_range), bounds=[o_range], method='Nelder-Mead')
                best_overlap, best_order = -res.fun, res.x[0]
        if best_overlap is None:
            best_order = orders[overlaps.argmax()]

        if self.visualizer:
            t1_norm = self.normalize_time(t1, to_norm_by1, best_order)
            t2_norm = self.normalize_time(t2, to_norm_by2, best_order)
            grid_coarse, scores_coarse = None, None
            if smooth_cost_function:
                grid_coarse, scores_coarse = orders, original_overlaps
            self.visualizer.visualize_grid_search(t1, t2, t1_norm, t2_norm, to_norm1, to_norm2, orders_cf, overlaps,
                                                  grid_coarse, scores_coarse)

        return best_order, overlaps, f

    def order_opt(self, rxn_1, rxn_2, spec_to_norm, spec_to_norm_by, o_range=None, method='Nelder-Mead'):
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
        rxn_1 : str
            Name of the first reaction.
        rxn_2 : str
            Name of the second reaction.
        spec_to_norm : str
            The name of the species whose trace should be normalized.
        spec_to_norm_by : str
            The name of the species whose trace should be integrated in order to normalize
            the time axis.
        o_range : tuple, optional
            The range of order values to search, as (min, max). Default None.
            These bounds will be ignored unless `method` is one of the follwing:
            "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", or "trust-constr-...".
        method : str, optional
            The minimization algorithm to use. Can take any of the values allowed
            by `scipy.optimize.minimize`. Default "Nelder-Mead".

        Returns
        -------
        best_o : float
            The reactant order that maximizes trace overlap
        rs : list
            A record of all the Pearson correlation coefficients for each
            of the orders tried, which are np.arange(o_range[0], o_range[1], o_step)

        """

        t1 = self.data.reaction_traces[rxn_1][:, 0]
        t2 = self.data.reaction_traces[rxn_2][:, 0]
        to_norm1 = self.data.reaction_traces[rxn_1][:, self.data.species_names.index(spec_to_norm) + 1]
        to_norm2 = self.data.reaction_traces[rxn_2][:, self.data.species_names.index(spec_to_norm) + 1]
        to_norm_by1 = self.data.reaction_traces[rxn_1][:, self.data.species_names.index(spec_to_norm_by) + 1]
        to_norm_by2 = self.data.reaction_traces[rxn_2][:, self.data.species_names.index(spec_to_norm_by) + 1]

        os = []
        overlaps = []

        # The ojective function to minimize
        def deviation(o):
            o = o[0]
            # a. normalize the time axis of both signals
            t1_norm = self.normalize_time(t1, to_norm_by1, o)
            t2_norm = self.normalize_time(t2, to_norm_by2, o)
            overlap = self.calculate_overlap(t1_norm, t2_norm, to_norm1, to_norm2, self.overlap_metric)
            os.append(o)
            overlaps.append(overlap)
            return -overlap

        # The minimization algorithm
        start_o = np.array([0])
        if o_range is not None:
            start_o = np.array([(o_range[0] + o_range[1]) / 2])
            o_range = [o_range]

        result = minimize(deviation, start_o, method=method, bounds=o_range)

        if self.visualizer:
            t1_norm = self.normalize_time(t1, to_norm_by1, result.x[0])
            t2_norm = self.normalize_time(t2, to_norm_by2, result.x[0])
            self.visualizer.visualize_opt(t1, t2, t1_norm, t2_norm, to_norm1, to_norm2, result, os, overlaps)

        return result, {'orders': os, 'overlaps': overlaps}

    def poison_search(self, rxn_1, rxn_2, spec_to_norm, spec_to_norm_by, order, poison_range=None, nsteps=100,
                      smooth_cost_function=True, win_cost=None, interp_cost_fun=True, window_type='blackman'):
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
        rxn_1 : str
            Name of the first reaction.
        rxn_2 : str
            Name of the second reaction.
        spec_to_norm : str
            The name of the species whose trace should be normalized.
        spec_to_norm_by : str
            The name of the species whose trace should be integrated in order to normalize
            the time axis.
        order : float, optional
            The assumed order of the reactant in the reaction rate law.
        poison_range : tuple, optional
            The range of poison concentrations to search
        nsteps : int, optional
            The number of different poison values in `poison_range` to try
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
        t1 = self.data.reaction_traces[rxn_1][:, 0]
        t2 = self.data.reaction_traces[rxn_2][:, 0]
        to_norm1 = self.data.reaction_traces[rxn_1][:, self.data.species_names.index(spec_to_norm) + 1]
        to_norm2 = self.data.reaction_traces[rxn_2][:, self.data.species_names.index(spec_to_norm) + 1]
        to_norm_by1 = self.data.reaction_traces[rxn_1][:, self.data.species_names.index(spec_to_norm_by) + 1]
        to_norm_by2 = self.data.reaction_traces[rxn_2][:, self.data.species_names.index(spec_to_norm_by) + 1]

        if poison_range is None:
            poison_range = (0, min([min(to_norm_by1), min(to_norm_by2)]) / 2)
        poisonings = np.linspace(poison_range[0], poison_range[1], nsteps)
        overlaps = np.zeros(poisonings.shape)
        for i, pois in enumerate(poisonings):
            # a. normalize the time axis of both signals
            t1_norm = self.normalize_time(t1, to_norm_by1 - pois, order)
            t2_norm = self.normalize_time(t2, to_norm_by2 - pois, order)
            overlaps[i] = self.calculate_overlap(t1_norm, t2_norm, to_norm1, to_norm2, self.overlap_metric)

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
            best_pois = poisonings[overlaps.argmax()]

        if self.visualizer:
            t1_norm = self.normalize_time(t1, to_norm_by1, order)
            t2_norm = self.normalize_time(t2, to_norm_by2, order)
            t1_norm_pois = self.normalize_time(t1, to_norm_by1 - best_pois, order)
            t2_norm_pois = self.normalize_time(t2, to_norm_by2 - best_pois, order)
            grid_coarse, scores_coarse = None, None
            if smooth_cost_function:
                grid_coarse, scores_coarse = poisonings, original_overlaps
            self.visualizer.visualize_grid_search(t1_norm, t2_norm, t1_norm_pois, t2_norm_pois, to_norm1, to_norm2,
                                                  poisonings_cf, overlaps, grid_coarse, scores_coarse)

        return best_pois, overlaps, f

    def poison_opt(self, rxn_1, rxn_2, spec_to_norm, spec_to_norm_by, order=None,
                   poison_range=None, method='Nelder-Mead'):
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
        rxn_1 : str
            Name of the first reaction.
        rxn_2 : str
            Name of the second reaction.
        spec_to_norm : str
            The name of the species whose trace should be normalized.
        spec_to_norm_by : str
            The name of the species whose trace should be integrated in order to normalize
            the time axis.
        order : float, optional
            The assumed order of the reactant in the reaction rate law.
        poison_range : tuple, optional
            The range of poison concentrations to search.
        method : str, optional
            The minimization algorithm to use. Can take any of the values allowed
            by `scipy.optimize.minimize`. Default "Nelder-Mead".

        Returns
        -------
        result : `scipy.OptimizeResult`
            The minimization result, with the optimal poisoning contained in result.x[0]
        history : dict
            A record of all the poisonings for each of the orders tried, {'poisonings': ps,
            'overlaps': overlaps}

        """
        t1 = self.data.reaction_traces[rxn_1][:, 0]
        t2 = self.data.reaction_traces[rxn_2][:, 0]
        to_norm1 = self.data.reaction_traces[rxn_1][:, self.data.species_names.index(spec_to_norm) + 1]
        to_norm2 = self.data.reaction_traces[rxn_2][:, self.data.species_names.index(spec_to_norm) + 1]
        to_norm_by1 = self.data.reaction_traces[rxn_1][:, self.data.species_names.index(spec_to_norm_by) + 1]
        to_norm_by2 = self.data.reaction_traces[rxn_2][:, self.data.species_names.index(spec_to_norm_by) + 1]

        ps = []
        overlaps = []

        # The ojective function to minimize
        def deviation(p):
            p = p[0]
            # a. normalize the time axis of both signals
            t1_norm = self.normalize_time(t1, to_norm_by1 - p, order)
            t2_norm = self.normalize_time(t2, to_norm_by2 - p, order)
            overlap = self.calculate_overlap(t1_norm, t2_norm, to_norm1, to_norm2, self.overlap_metric)
            ps.append(p)
            overlaps.append(overlap)
            return -overlap

        # The minimization algorithm
        if poison_range is None:
            poison_range = (0, min([min(to_norm_by1), min(to_norm_by2)]) / 2)
        start_p = np.mean(poison_range)

        result = minimize(deviation, start_p, method=method, bounds=[poison_range])

        if self.visualizer:
            t1_norm = self.normalize_time(t1, to_norm_by1, result.x[0])
            t2_norm = self.normalize_time(t2, to_norm_by2, result.x[0])
            t1_norm_pois = self.normalize_time(t1, to_norm_by1 - result.x[0], result.x[0])
            t2_norm_pois = self.normalize_time(t2, to_norm_by2 - result.x[0], result.x[0])
            self.visualizer.visualize_opt(t1_norm, t2_norm, t1_norm_pois, t2_norm_pois, to_norm1, to_norm2,
                                          result, ps, overlaps)

        return result, {'poisonings': ps, 'overlaps': overlaps}

    def order_poison_opt(self, spec_to_norm, spec_to_norm_by, rxns=None, o_range=None, poison_range=None,
                         method='Nelder-Mead'):
        """
        Returns the optimal order and poisoning for a reactant in a rate law using the VTNA procedure.

        Given 3+ reaction traces, each consisting of a reactant trace and
        a product trace, where the reactant concentration differed between
        the traces, compute the order of the reactant in the reaction
        rate law and the amount by which it was poisoned. This order,
        when used to normalize the time for each reaction trace by the
        reactant concentration, will cause the two reaction traces (product
        or reactant traces) to overlay onto each other. This function maximizes
        the overlap between the time normalized and poison-adjusted reaction traces.

        Parameters
        ----------
        rxns : list of str
            The names of the reactions to use for fitting. Must contain at least 3 reactions
            with different starting concentrations of `spec_to_norm_by`.
        spec_to_norm : str
            The name of the species whose trace should be normalized.
        spec_to_norm_by : str
            The name of the species whose trace should be integrated in order to normalize
            the time axis.
        o_range : tuple, optional
            The range of order values to search, as (min, max). Default None.
            These bounds will be ignored unless `method` is one of the follwing:
            "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", or "trust-constr-...".
        poison_range : tuple, optional
            The range of poisoning values to search, as (min, max). Default None.
            These bounds will be ignored unless `method` is one of the follwing:
            "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", or "trust-constr-...".
        method : str, optional
            The minimization algorithm to use. Can take any of the values allowed
            by `scipy.optimize.minimize`. Default "Nelder-Mead".

        Returns
        -------
        result : `scipy.OptimizeResult`
            The minimization result, with the optimal order and poisoning contained in
            result.x[0] and result.x[1], respectively
        history : dict
            A record of all the orders and poisonings tried, {'orders': os, 'poisonings': ps,
            'overlaps': overlaps}
        """
        if rxns is None:
            rxns = self.data.reaction_names
        ts = [self.data.reaction_traces[rxn][:, 0] for rxn in rxns]
        to_norms = [self.data.reaction_traces[rxn][:, self.data.species_names.index(spec_to_norm) + 1]
                    for rxn in rxns]
        to_norm_bys = [self.data.reaction_traces[rxn][:, self.data.species_names.index(spec_to_norm_by) + 1]
                       for rxn in rxns]

        os = []
        ps = []
        overlaps = []

        # The ojective function to minimize
        def deviation(x):
            o = x[0]
            p = x[1]
            overlap = 0
            for (idx_1, idx_2) in itertools.combinations(range(len(ts)), 2):
                t1, t2 = ts[idx_1], ts[idx_2]
                to_norm_by1, to_norm_by2 = to_norm_bys[idx_1], to_norm_bys[idx_2]
                to_norm1, to_norm2 = to_norms[idx_1], to_norms[idx_2]
                t1_norm = self.normalize_time(t1, to_norm_by1 - p, o)
                t2_norm = self.normalize_time(t2, to_norm_by2 - p, o)
                overlap += self.calculate_overlap(t1_norm, t2_norm, to_norm1, to_norm2, self.overlap_metric)
            os.append(o)
            ps.append(p)
            overlaps.append(overlap)
            return -overlap

        # The minimization algorithm
        start_o = np.array([0])
        if o_range is not None:
            start_o = np.mean(o_range)

        # The minimization algorithm
        if poison_range is None:
            poison_range = (0, min([min(tnb) for tnb in to_norm_bys]) / 2)
        start_p = np.mean(poison_range)
        result = minimize(deviation, np.array([start_o, start_p]), method=method,
                          bounds=[o_range, poison_range])

        if self.visualizer:
            t_norms = [self.normalize_time(t, tnb - result.x[1], result.x[0])
                       for (t, tnb) in zip(ts, to_norm_bys)]
            self.visualizer.visualize_2d_opt(ts, t_norms, to_norms, result, os, ps, overlaps)

        return result, {'orders': os, 'poisonings': ps, 'overlaps': overlaps}

    def order_search_single_trace(self, rxn, spec_to_norm, spec_to_norm_by, o_range=(0, 3), nsteps=100,
                                  handle_neg=None, smooth_cost_function=False, win_cost=None, interp_cost_fun=False,
                                  window_type='blackman'):
        """
        Returns the optimal order for a reactant in a rate law using the VTNA procedure.

        Given a single reaction traces, compute the order of the reactant in the reaction
        rate law. This order, when used to normalize the time for the `spec_to_norm`
        reaction trace by the concentration of the species `to_norm_by`, will cause the
        reaction trace to become linear. This function maximizes the R^2 score (closest to
        1 or -1)  of the best fit line to the normalized trace.

        Parameters
        ----------
        rxn : str
            Name of the reaction.
        spec_to_norm : str
            The name of the species whose trace should be normalized.
        spec_to_norm_by : str
            The name of the species whose trace should be integrated in order to normalize
            the time axis.
        o_range : tuple, optional
            The range of order values to search, as (min, max). Default None.
            These bounds will be ignored unless `method` is one of the follwing:
            "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", or "trust-constr-...".
            However, the mean of the range will still be used as a starting point.
        nsteps : int, optional
            The number of different poison values in `poison_range` to try
        handle_neg : function, optional
            A function that manipulates negative values in a reaction trace in place,
            such as by replacing them with another value. The function should have the
            signature (arr: numpy.ndarray) -> None
        smooth_cost_function : bool, Optional
            If True, the cost function estimated by the grid search will
            be smoothed before determining the optimal order. Useful
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

        Returns
        -------
        result : `scipy.OptimizeResult`
            The minimization result, with the optimal order contained in result.x[0]
        k : float
            The rate constant, given by the slope of the normalized line
        history : dict
            A record of all the orders for each of the orders tried, {'orders': ps,
            'costs': overlaps}

        """
        t = self.data.reaction_traces[rxn][:, 0]
        to_norm = self.data.reaction_traces[rxn][:, self.data.species_names.index(spec_to_norm) + 1]
        to_norm_by = self.data.reaction_traces[rxn][:, self.data.species_names.index(spec_to_norm_by) + 1]
        to_norm = to_norm.reshape(-1, 1)  # necessary for sklearn's linear regression
        if handle_neg:
            to_norm_by = copy.deepcopy(to_norm_by)
            handle_neg(to_norm_by)

        orders = np.linspace(o_range[0], o_range[1], nsteps)
        scores = np.zeros(orders.shape)
        for i, order in enumerate(orders):
            t_norm = VTNA.normalize_time(t, to_norm_by, order)
            # print(t[np.isnan(t_norm)], to_norm_by[np.isnan(t_norm)])
            t_norm = t_norm.reshape(-1, 1)

            # perform linear regression and calculate R^2
            line = LinearRegression()
            line.fit(t_norm, to_norm)
            scores[i] = line.score(t_norm, to_norm)

        original_scores = scores
        orders_cf = orders
        best_score, best_order = None, None
        f = None
        if smooth_cost_function:
            if win_cost is None:
                win_cost = len(scores) // 2
            overlaps = smooth(scores, window_len=win_cost, window=window_type, general_sig=True)
            if interp_cost_fun:
                f = interp1d(orders, -scores, kind='quadratic')
                orders_cf = np.linspace(o_range[0], o_range[1], nsteps * 10)
                scores = -f(orders_cf)
                res = minimize(f, np.mean(o_range), bounds=[o_range], method='Nelder-Mead')
                best_score, best_order = -res.fun, res.x[0]
        if best_score is None:
            best_order = orders[scores.argmax()]

        _, line = self.calculate_linearity(t, to_norm, to_norm_by, best_order)
        k = line.coef_[0][0]

        if self.visualizer:
            t_norm = self.normalize_time(t, to_norm_by, best_order)
            grid_coarse, scores_coarse = None, None
            if smooth_cost_function:
                grid_coarse, scores_coarse = orders, original_scores
            self.visualizer.visualize_grid_search_single_trace(t, t_norm, to_norm, orders_cf, scores, line,
                                                               grid_coarse, scores_coarse)

        return best_order, scores, f, k

    def order_opt_single_trace(self, rxn, spec_to_norm, spec_to_norm_by, o_range=(0, 3), method='Nelder-Mead',
                               handle_neg=None, **kwargs):
        """
        Returns the optimal order for a reactant in a rate law using the VTNA procedure.

        Given a single reaction traces, compute the order of the reactant in the reaction
        rate law. This order, when used to normalize the time for the `spec_to_norm`
        reaction trace by the concentration of the species `to_norm_by`, will cause the
        reaction trace to become linear. This function maximizes the R^2 score (closest to
        1 or -1)  of the best fit line to the normalized trace.

        Parameters
        ----------
        rxn : str
            Name of the reaction.
        spec_to_norm : str
            The name of the species whose trace should be normalized.
        spec_to_norm_by : str
            The name of the species whose trace should be integrated in order to normalize
            the time axis.
        o_range : tuple, optional
            The range of order values to search, as (min, max). Default None.
            These bounds will be ignored unless `method` is one of the follwing:
            "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", or "trust-constr-...".
            However, the mean of the range will still be used as a starting point.
        method : str, optional
            The minimization algorithm to use. Can take any of the values allowed
            by `scipy.optimize.minimize`. Default "Nelder-Mead".
        handle_neg : function, optional
            A function that manipulates negative values in a reaction trace in place,
            such as by replacing them with another value. The function should have the
            signature (arr: numpy.ndarray) -> None

        Returns
        -------
        result : `scipy.OptimizeResult`
            The minimization result, with the optimal order contained in result.x[0]
        k : float
            The rate constant, given by the slope of the normalized line
        history : dict
            A record of all the orders for each of the orders tried, {'orders': ps,
            'costs': overlaps}

        """
        t = self.data.reaction_traces[rxn][:, 0]
        to_norm = self.data.reaction_traces[rxn][:, self.data.species_names.index(spec_to_norm) + 1]
        to_norm_by = self.data.reaction_traces[rxn][:, self.data.species_names.index(spec_to_norm_by) + 1]

        if handle_neg:
            to_norm_by = copy.deepcopy(to_norm_by)
            handle_neg(copy.deepcopy(to_norm_by))

        k = []
        orders = []
        costs = []

        def cost_function(order):
            cost, line = self.calculate_linearity(t, to_norm, to_norm_by, order[0])
            k.append(line.coef_[0][0])
            orders.append(order)
            costs.append(cost)
            return cost

        opt_res = minimize(cost_function, np.mean(o_range), bounds=[o_range], method=method, **kwargs)

        if self.visualizer:
            t_norm = self.normalize_time(t, to_norm_by, opt_res.x[0])
            _, line = self.calculate_linearity(t, to_norm, to_norm_by, opt_res.x[0])
            self.visualizer.visualize_opt_single_trace(t, t_norm, to_norm, opt_res, line, orders, costs)

        return opt_res, k[-1], {'orders': orders, 'costs': costs}
