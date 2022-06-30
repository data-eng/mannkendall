# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 MeteoSwiss, contributors of the original matlab version of the code listed in
ORIGINAL_AUTHORS.
Copyright (c) 2020 MeteoSwiss, contributors of the Python version of the code listed in AUTHORS.

Distributed under the terms of the BSD 3-Clause License.

SPDX-License-Identifier: BSD-3-Clause

This file contains the core statistical routines for the package.
"""

# Import the required packages
from datetime import datetime
import numpy as np
from scipy.interpolate import interp1d
import scipy.stats

# Import from this package
from . import mk_tools as mkt

def std_normal_var(s, var_s):
    """ Compute the normal standard variable Z.

    From Gilbert (1987).

    Args:
        s (int): S statistics of the Mann-Kendall test computed from the S_test.
        k_var (float): variance of the time series taking into account the ties in values and time.
                       It should be computed by Kendall_var().

    Returns:
        float: S statistic weighted by the variance.

    """

    # First some sanity checks.
    # Be forgiving if I got a float ...
    if isinstance(s, float) and s.is_integer():
        s = int(s)
    if not isinstance(s, (int)):
        raise Exception('Ouch ! Variable s must be of type int, not: %s' % (type(s)))
    if not isinstance(var_s, (int, float)):
        raise Exception('Ouch ! Variable var_s must be of type float, not: %s' % (type(s)))

    # Deal with the case when s is 0
    if s == 0:
        return 0.0

    # Deal with the other cases.
    return (s - np.sign(s))/var_s**0.5

def sen_slope( obs, k_var, alpha_cl=90., method='brute' ):
    """ Compute Sen's slope.

    Specifically, this computes the median of the slopes for each interval:

        (xj-xi)/(j-i), j>i

    Args:
        obs (2D ndarray of floats): the data array. The first column is
        MATLAB timestamps and the second column is the observations.

        k_var (float): Kendall variance, computed with Kendall_var.

        confidence (float, optional): the desired confidence limit, in %. Defaults to 90.

        method (string, opt): Method for calculating slope. One of:
                              "siegel", "thiel": as implemented in scipy.
                              "brute" (default): builds the n x n array of slopes and sorts it
                              to find the median and the confidence limits.
                              "brute-sparse": same as brute, but also computes confidence limits
                              with an interpolation. When datapoints are few.

    Return:
        (float, float, float): Sen's slope, lower confidence limit, upper confidence limit.

    Note:
        The slopes are returned in units of 1/s.
    """

    # Start with some sanity checks
    if not isinstance(alpha_cl, (float, int)):
        raise Exception('Ouch! confidence should be of type int, not: %s' % (type(alpha_cl)))
    if alpha_cl > 100 or alpha_cl < 0:
        raise Exception('Ouch ! confidence must be 0<=alpha_cl<=100, not: %f' % (float(alpha_cl)))
    if not isinstance(k_var, (int, float)):
        raise Exception('Ouch ! The variance must be of type float, not: %s' % (type(k_var)))

    (cols,rows) = obs.shape
    if cols != 2:
        raise Exception( "There must be two columns in obs" )

    if method == "siegel":
        # TODO: write an iterative NaN remover
        obsT = obs.T
        good = ((obsT)[~np.isnan(obsT).any(axis=1)]).T
        (slope,intercept) = scipy.stats.siegelslopes( good[1,:], good[0,:], method='separate' )
        #(slope,intercept) = scipy.stats.siegelslopes( good[1,:], good[0,:], method='hierarchical' )
        lcl = 0 # how will these be computed?
        ucl = 0 # how will these be computed?
    elif method == "theil":
        # TODO: write an iterative NaN remover
        obsT = obs.T
        good = ((obsT)[~np.isnan(obsT).any(axis=1)]).T
        a = float(alpha_cl) / 100
        (slope,intercept,lcl,ucl) = scipy.stats.theilslopes( good[1,:], good[0,:], alpha=a, method='joint' )
        #(slope,intercept,lcl,ucl) = scipy.stats.theilslopes( good[1,:], good[0,:], alpha=a, method='separate' )
    else:
        # Let's compute the slope for all the possible pairs.
        d = np.array([item for i in range(0, rows-1)
                      for item in list((obs[1,i+1:rows] - obs[1,i])/mkt.days_to_s(obs[0,i+1:rows] - obs[0,i]))])

        # Only keep valid values
        d = d[~np.isnan(d)]
        # Sort
        # This will get us the median, and is
        # also needed for interpolation below
        d.sort()

        l = len(d)
        if l % 2 == 1:
            slope = d[(l-1)//2]
            # these m_1, m_2 defaults will be overriden below
            # unless cconf is very low
            m_1 = (l-1)//2 - 1
            m_2 = (l-1)//2 + 1
        else:
            slope = (d[l//2-1]+d[l//2])/2
            # these m_1, m_2 defaults will be overriden below
            # unless cconf is very low
            m_1 = l//2 - 2
            m_2 = l//2 + 1

        # Apply the confidence limits
        kvarroot = k_var**0.5
        if np.isnan(kvarroot):
            # if k_var is small, the sqrt is a NaN and a RuntimeWarning is issued.
            # For such low cconf, default to the values on either side of the median.
            cconf = 0.0
            # Keep the default m_1, m_2 values from above
        else:
            cconf = -scipy.stats.norm.ppf((1-alpha_cl/100)/2) * kvarroot
            # Note: because python starts at 0 and not 1, we need an additional "-1" to
            # the following values of m_1 and m_2 to match the matlab implementation.
            m_1 = (0.5 * (len(d) - cconf)) - 1
            m_2 = (0.5 * (len(d) + cconf)) - 1


        if method == "brute-sparse":
            # Interpolate when datapoints are sparse
            f = interp1d(np.arange(0, l, 1), d, kind='linear',
                         fill_value=(d[0],d[-1]), assume_sorted=True, bounds_error=False)
            lcl = f(m_1)
            ucl = f(m_2)
        else:
            lcl = d[int(m_1)]
            ucl = d[int(m_2)]

    return (float(slope), float(lcl), float(ucl))

def s_test( obs ):
    """ Compute the S statistics (Si) for the Mann-Kendall test.

    From Gilbert (1987).

    Args:
        obs (2D ndarray<float>): the data array. The first column is MATLAB timestamps
                                 and the second column is the observations.
                                 MUST be time-ordrered.

    Returns:
        (float, ndarray): S, n.
                          S (float) = double sum on the sign of the difference between data pairs
                          (Si).
                          n (ndarray of int) = number of valid data in each year of the time series

    """

    # Some sanity checks first
    if not isinstance(obs, np.ndarray):
        raise Exception('Ouch ! I was expecting some numpy.ndarray, not: %s' % (type(item)))

    # Find the limiting years
    obs_years = np.array([mkt.mat2datetime(item).year for item in obs[0,:]])
    min_year = np.min(obs_years)
    max_year = np.max(obs_years)

    # An array to keep track of the number of valid data points in each season
    n = np.zeros(max_year - min_year + 1) * np.nan
    # Create a vector to keep track of the results
    sij = np.zeros(max_year - min_year + 1) * np.nan

    for (yr_ind, year) in enumerate(range(min_year, max_year+1)):
        #How valid points do I have :
        n[yr_ind] = np.count_nonzero(~np.isnan((obs[1,:])[obs_years == year]))

        # Compute s for that year, by summing the signs for the differences with all the upcoming
        # years
        sij[yr_ind] = np.nansum([np.sign(item - (obs[1,:])[obs_years == year])
                                 for yr2 in range(year+1, max_year+1)
                                 for item in (obs[1,:])[obs_years == yr2]])

    return (np.nansum(sij), n)
