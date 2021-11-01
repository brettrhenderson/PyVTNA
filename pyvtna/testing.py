import numpy as np


def simulate_reaction_trace(time, reac_st, reac_orders, reac_coefs, prod_st, prod_coefs, k=1, noise=0.0):
    dts = time[1:] - time[:-1]
    reacs = [[reac_st[i]] for i in range(len(reac_st))]
    prods = [[prod_st[i]] for i in range(len(prod_st))]

    # simulate the reaction
    for i, dt in enumerate(dts):
        # calculate the rate
        rate = k
        for (reac, order) in zip(reacs, reac_orders):
            rate *=reac[i]**order

        # calculate the amount of each species
        for (reac, coef) in zip(reacs, reac_coefs):
            reac.append(reac[i] - dt * coef * rate)
        for (prod, coef) in zip(prods, prod_coefs):
            prod.append(prod[i] + dt * coef * rate)
    reacs = [np.array(reac) for reac in reacs]
    prods = [np.array(prod) for prid in prods]

    if hasattr(noise, '__iter__'):
        for (n, spec) in zip(noise, reacs + prods):
            spec += np.random.normal(0, n, len(time))
    else:
        for spec in reacs + prods:
            spec += np.random.normal(0, noise, len(time))
    return reacs, prods

def make_sigs(sig2dil, st1, e1, st2, e2, ascending=True, freq=0.01, noise=0.03, tau=2.0):
    """
    Create two test signals of the form 1 - exp((t-t0) / tau) or exp(-(t-t0) / tau).
    
    The second signal will have the same form as the first but be dilated in the x-direction
    by the specified amount. Used for testing the singal processing functions.
    
    Parameters
    ----------
    sig2dil: float
        The amount by which to dilate the second signal relative to the first.
    st1: float
        The time at which the first signal starts (it will have either a 0 or 1 baseline before this,
        depending on whether the signal is ascending or descending).
    e1: float
        The end time of the first signal.
    st2: float
        The time at which the second signal starts (it will have either a 0 or 1 baseline before this,
        depending on whether the signal is ascending or descending).
    e2: float
        The end time of the second signal.
    ascending: bool, optional
        Whether the signals should be ascending (if False, descending). Default True.
    freq: float, optional
        The sampling frequency of the signal. Default 0.01.
    noise: float, optional
        Standard deviation of Gaussian noise to be added to the signals. Default 0.03. 
    tau: float, optional
        The time constant of the exponential signals. Default 2.0.
    
    """
    freq = freq    # sampling frequency (constant for mass spec signals but may vary for NMR, etc.)
    sig2_dilation = sig2dil   # amount signal 2 is stretched by relative to signal 1

    # SIGNAL 1
    # capture starts at 0, but signal doesn't begin until later
    # the start and end times for signal 1
    t1_start, t1_end = st1, e1
    t1_start_idx = int(t1_start / freq)
    t1 = np.arange(0, t1_end, freq)
    if ascending:
        sig1 = 1 - np.exp(-(t1 - t1_start) / tau)
        sig1[:t1_start_idx] = 0
    else:
        sig1 = np.exp(-(t1 - t1_start) / tau)
        sig1[:t1_start_idx] = 1

    # SIGNAL 2
    # the start and end times for signal 2
    t2_start, t2_end = st2, e2
    t2_start_idx = int(t2_start / freq)
    t2 = np.arange(0, t2_end, freq)

    # signal 2
    if ascending:
        sig2 = 1 - np.exp(-(t2 - t2_start) / (tau * sig2_dilation))
        sig2[:t2_start_idx] = 0
    else:
        sig2 = np.exp(-(t2 - t2_start) / (tau * sig2_dilation))
        sig2[:t2_start_idx] = 1

    # add noise 
    sig1_noisy = sig1 + np.random.normal(0,noise,len(sig1))
    sig2_noisy = sig2 + np.random.normal(0,noise,len(sig2))
    s1 = np.concatenate([t1.reshape(-1, 1), sig1_noisy.reshape(-1, 1)], axis=1)
    s2 = np.concatenate([t2.reshape(-1, 1), sig2_noisy.reshape(-1, 1)], axis=1)
    return s1, s2