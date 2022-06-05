# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 MeteoSwiss, contributors of the original matlab version of the code listed in
ORIGINAL_AUTHORS.
Copyright (c) 2020 MeteoSwiss, contributors of the Python version of the code listed in AUTHORS.

Distributed under the terms of the BSD 3-Clause License.

SPDX-License-Identifier: BSD-3-Clause

This file contains the core routines for the mannkendall package.
"""

# Import python packages
import warnings
import numpy as np
import scipy.stats as spstats

# Import from this package
from . import mk_hardcoded as mkh
from . import mk_tools as mkt
from . import mk_stats as mks
from . import mk_white as mkw


def prob_3pw(p_pw, p_tfpw_y, alpha_mk):
    """ Estimate the probability of the MK test and its statistical significance.

        1) Estimates the probability of the MK test with the 3PW method. To have the maximal
           certainty, P is taken as the maximum of P_PW and P_TFPW_Y.
        2) Estimates the statistical significance of the MK test as a function of the given
           confidence level alpha_MK.

    Args:
        p_pw (float): probability computed from the PW prewhitened dataset
        p_tfpw_y (float): probability computed from the TFPW_Y prewhitened dataset
        alpha_mk (float): confidence level in % for the MK test.

    Returns:
        (float, int): P, ss

    Todo:
        * improve this docstring

    """

    # Some sanity checks to begin with
    for item in [p_pw, p_tfpw_y, alpha_mk]:
        if not isinstance(item, (int, float)):
            raise Exception('Ouch ! I was expecting a float, not: %s' % (type(item)))

    p_alpha = 1 - alpha_mk/100

    # Compute the probability
    p = np.nanmax([p_pw, p_tfpw_y])

    # Determine the statistical significance
    if (p_pw <= p_alpha) & (p_tfpw_y <= p_alpha):
        ss = alpha_mk
    elif (p_pw > p_alpha) & (p_tfpw_y <= p_alpha): # false positive for TFPW_Y @ alpha %
        ss = -1
    elif (p_tfpw_y > p_alpha) & (p_pw <= p_alpha): # false positive for TFPW_Y
        ss = -2
    elif (p_tfpw_y > p_alpha) & (p_pw > p_alpha): # false positive for PW
        ss = 0

    return (p, ss)


def compute_mk_stat(obs, resolution, alpha_mk=95, alpha_cl=90):
    """ Compute all the components for the MK statistics.

    Args:
        obs (2D ndarray of floats): the data array. The first column is
        MATLAB timestamps and the second column is the observations.
        resolution (float): delta value below which two measurements are considered equivalent.
        alpha_mk (float, optional): confidence level for the Mann-Kendall test in %. Defaults to 95.
        alpha_cl (float, optional): confidence level for the Sen's slope in %. Defaults to 90.

    Returns:
        (dict, int, float, float): result, s, vari, z

    """

    # Some sanity checks
    for item in [alpha_mk, alpha_cl]:
        if not isinstance(item, (int, float)):
            raise Exception('Ouch! alphas must be of type float, not: %s' %(type(item)))
    if alpha_mk < 0 or alpha_mk > 100 or alpha_cl < 0 or alpha_cl > 100:
        raise Exception("Ouch ! Confidence limits must be 0 <= CL <= 100.")

    result = {}

    ## temp hack, until float timestamps are pushed all the way down
    tt = []
    for t in obs[0,:]:
        tt.append( mkt.mat2datetime(t) ) 
    obs_dts = np.array(tt)

    t = mkt.nb_tie(obs[1,:], resolution)
    (s, n) = mks.s_test(obs[1,:], obs_dts)
    vari = mkt.kendall_var(obs[1,:], t, n)
    z = mks.std_normal_var(s, vari)

    (cols,rows) = obs.shape
    if rows > 10:
        result['p'] = 2 * (1 - spstats.norm.cdf(np.abs(z), loc=0, scale=1))
    else:
        prob_mk_n = mkh.PROB_MK_N
        result['p'] = prob_mk_n[np.abs(s), cols] # TODO: np.abs(s) + 1 ?

    # Determine the statistic significance
    if result['p'] <= 1- alpha_mk/100:
        result['ss'] = alpha_mk
    else:
        result['ss'] = 0

    (slope, slope_min, slope_max) = mks.sen_slope(obs, vari, alpha_cl=alpha_cl)
    # Transform the slop in 1/yr.
    result['slope'] = slope * 3600 * 24 * 365.25
    result['ucl'] = slope_max * 3600 * 24 *365.25
    result['lcl'] = slope_min * 3600 * 24 * 365.25

    return (result, s, vari, z)

