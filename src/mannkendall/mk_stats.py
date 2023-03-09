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
import heapq
from datetime import datetime
from contextlib import ExitStack
import copy
import os
import subprocess
import tempfile
import uuid

import numpy as np
from scipy.interpolate import interp1d
import scipy.stats

# Import from this package
from . import mk_tools as mkt
from . import median_slope_bins as bins

import psycopg2


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

def sen_slope( obs, k_var, alpha_cl=90., method='brute-disk' ):
    """ Compute Sen's slope.

    Specifically, this computes the median of the slopes for each interval:

        (xj-xi)/(j-i), j>i

    Args:
        obs (2D ndarray of floats): the data array. The first column is
        MATLAB timestamps and the second column is the observations.

        k_var (float): Kendall variance, computed with Kendall_var.

        confidence (float, optional): the desired confidence limit, in %. Defaults to 90.

    Return:
        (float, float, float): Sen's slope, lower confidence limit, upper confidence limit.

    Note:
        The slopes are returned in units of 1/s.
    """

    """
    method (string, opt): Method for calculating slope. One of:
                              "brute" (default): builds the n(n-1)/2 array of slopes and sorts
                              it to find the median and the confidence limits.
                              "brute-sparse": same as brute, but computes confidence limits
                              with an interpolation. Use when datapoints are few.
                              "thiel": as implemented in scipy.
                              "bins": Estimates the slope distributions and uses this estimate
                              to only build three small parts of the complete n(n-1)/2 array of
                              slopes, thoses that contain the median slope and the lcl, ucl.
                              "brute-disk": same as brute but instead of building the numpy array of slopes, 
                              writes the computed values in a file under TMPDIR, sorts the file using bash sort 
                              and gets the median using bash sed.
                              "brute-disk": same as brute but instead of building the numpy array of slopes,
                              writes the computed values in a file under TMPDIR and sorts the file using bash sort
                              "brute-sparse": same as brute, but also computes confidence limits
                              with an interpolation. When datapoints are few.
    """
    method = os.getenv('SEN_SLOPE_METHOD', 'brute')

    # Start with some sanity checks
    if not isinstance(alpha_cl, (float, int)):
        raise Exception('Ouch! confidence should be of type int, not: %s' % (type(alpha_cl)))
    if alpha_cl > 100 or alpha_cl < 0:
        raise Exception('Ouch ! confidence must be 0<=alpha_cl<=100, not: %f' % (float(alpha_cl)))
    if not isinstance(k_var, (int, float)):
        raise Exception('Ouch ! The variance must be of type float, not: %s' % (type(k_var)))

    alpha = float(alpha_cl) / 100

    # NaN removal
    #obsT = obs.T
    #good1 = ((obsT)[~np.isnan(obsT).any(axis=1)]).T
    good_values = ~np.isnan(obs[1])
    good2 = np.array([ obs[0][good_values], obs[1][good_values] ])
    obs = copy.copy(good2)
    obs[0] = mkt.days_to_s(obs[0])

    (cols,rows) = obs.shape
    if cols != 2:
        raise Exception( "There must be two columns in obs" )

    # Besides the median, we will also need the confidence limits.
    # Here we calculate the indexes for the lcl and the ucl

    # The num of slopes is Sum rows-1, rows-2, ... 1 = rows*(rows-1)/2
    l = rows*(rows-1)/2
    if l % 2 == 1:
        slope_idx_1 = int( (l-1)//2 )
        slope_idx_2 = int( (l-1)//2 )
        # these m_1, m_2 defaults will be overriden below,
        # unless k_var is very low
        m_1 = (l-1)//2 - 1
        m_2 = (l-1)//2 + 1
    else:
        slope_idx_1 = int( l//2-1 )
        slope_idx_2 = int( l//2 )
        # these m_1, m_2 defaults will be overriden below,
        # unless k_var is very low
        m_1 = l//2 - 2
        m_2 = l//2 + 1

    # Apply the confidence limits
    kvarroot = k_var**0.5
    if np.isnan(kvarroot):
        # if k_var is small, the sqrt is a NaN and a RuntimeWarning is issued.
        # cconf is effectively zero, so m_1 and m_2 are the same as the median.
        # In this case, default to the values on either side of the median to
        # ensure m_1 < median < m_2
        cconf = 0.0
        # Keep the default m_1, m_2 values from above
    else:
        cconf = -scipy.stats.norm.ppf((1-alpha)/2) * kvarroot
        # Note: because python starts at 0 and not 1, we need an additional "-1" to
        # the following values of m_1 and m_2 to match the matlab implementation.
        m_1 = (0.5 * (l - cconf)) - 1
        m_2 = (0.5 * (l + cconf)) - 1


    if method == "theil":
        a = float(alpha_cl) / 100
        (slope,intercept,lcl,ucl) = scipy.stats.theilslopes( obs[1,:], obs[0,:], alpha=alpha )


    elif method == "bins":
        d = bins.initializer( obs, slope_idx_1, slope_idx_2, m_1, m_2 )
        is_good = bins.populate_bins( d )
        if not is_good:
            rebalanced = True
            while rebalanced:
                r = bins.rebalance( d )
                if (r is None):
                    rebalanced = False
                else:
                    (low_bin, mid_bin, high_bin) = bins.recount_bins( d )
        bins.populate_bins( d )
        (lcl,slope,ucl) = bins.get_percentiles( d )


    elif method == "postgres":

        user = os.getenv('POSTGRES_USER')
        password = os.getenv('POSTGRES_PASS')
        host = os.getenv('POSTGRES_HOST')
        port = os.getenv('POSTGRES_PORT')
        dbname = os.getenv('POSTGRES_DB')

        db_connect_kwargs = {
            'dbname': dbname,
            'user': user,
            'password': password,
            'host': host,
            'port': port
        }

        conn = None

        try:

            # connect to the PostgreSQL server
            conn = psycopg2.connect(**db_connect_kwargs)
            conn.set_session(autocommit=True)


            # create a cursor
            cursor = conn.cursor()

            # execute a statement
            cursor.execute('CREATE TEMPORARY TABLE temp_obs(time DOUBLE PRECISION, obs DOUBLE PRECISION')

            # display the result
            db_version = cursor.fetchone()
            print(db_version)

            # close the communication with the PostgreSQL
            cursor.close()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

        finally:
            if conn is not None:
                conn.close()
                print('Database connection closed.')




    elif method == "brute-disk":

        slopes_dir = os.path.join(tempfile.gettempdir(), 'slopes_' + str(uuid.uuid4()))

        os.makedirs(slopes_dir)

        tmp_array_l = int(os.getenv('TMP_ARRAY_L', 400_000_000))
        tmp_array_last_index = tmp_array_l - 1

        tmp_array = np.empty(tmp_array_l)

        tmp_files = []

        tmp_array_c = 0

        # compute the slopes
        for i in range(0, rows-1):
            for j in range(i + 1, rows):

                val = (obs[1, j] - obs[1, i]) / (obs[0, j] - obs[0, i])

                tmp_array[tmp_array_c] = val

                if tmp_array_c == tmp_array_last_index:
                    tmp_array.sort()
                    out_file = slopes_dir + os.sep + str(uuid.uuid4())
                    np.savetxt(out_file, tmp_array)
                    tmp_files.append(out_file)
                    tmp_array_c = 0
                else:
                    tmp_array_c += 1

        if tmp_array_c != 0:  # write remaining values
            tmp_array = tmp_array[:tmp_array_c]
            tmp_array.sort()
            out_file = slopes_dir + os.sep + str(uuid.uuid4())
            np.savetxt(out_file, tmp_array)
            tmp_files.append(out_file)

        if l % 2 == 1:
            median_pos = l // 2
            is_even = False
        else:
            median_pos = (l - 1) // 2
            median_pos_2 = median_pos + 1
            is_even = True

        m_1_pos = int(m_1)
        m_2_pos = int(m_2)

        break_limit = max([median_pos, median_pos+1, m_1_pos, m_2_pos])

        with ExitStack() as stack:

            files = [stack.enter_context(open(fname, 'r')) for fname in tmp_files]

            line_num = 0

            for line in heapq.merge(*files, key=float):

                if line_num > break_limit:
                    break
                if line_num == m_1_pos:
                    lcl = float(line)
                if line_num == median_pos:
                    slope = float(line)
                elif is_even and line_num == median_pos_2:
                    slope = (float(slope) + float(line)) / 2
                if line_num == m_2_pos:
                    ucl = float(line)
                line_num += 1

        for f in tmp_files:
            os.remove(f)
        # TODO remove the temporary directory

    else:
        # Make an array with all the possible pairs.
        ll = len(obs[1])
        d = np.array([item for i in range(0, ll-1)
                      for item in list((obs[1][i+1:ll] - obs[1][i])/(obs[0][i+1:ll] - obs[0][i]))])
        d.sort()
        l = len(d)
        if l % 2 == 1:
            slope = d[(l-1)//2]
        else:
            slope = (d[l//2-1]+d[l//2])/2

        if method == "brute-sparse":
            # Interpolate when datapoints are sparse
            f = interp1d(np.arange(0, l, 1), d, kind='linear',
                         fill_value=(d[0],d[-1]), assume_sorted=True, bounds_error=False)
            lcl = f(m_1)
            ucl = f(m_2)
        else:
            lcl = d[int(m_1)]
            ucl = d[int(m_2)]

    return (slope, lcl, ucl)

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
