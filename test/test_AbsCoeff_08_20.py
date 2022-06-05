# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 NCSR "Demokritos"

Distributed under the terms of the BSD 3-Clause License.

SPDX-License-Identifier: BSD-3-Clause

This file contains test functions
"""

import sys
import numpy
import datetime

import mannkendall as mk
from mannkendall import mk_stats as mks
from mannkendall import mk_white as mkw
from mannkendall import mk_tools as mkt


flat_tt = []
flat_vv = []

basename = sys.argv[1] 

f = open( basename + ".csv", "r" )
while True:
    line = f.readline()
    if not line: break
    a = line.split( " " )

    # Matlab time is days since 1/1/0000.
    # Python ordinal time is since 1/1/0001.
    datenum = float(a[0])
    days = datenum % 1
    t = datetime.datetime.fromordinal(int(datenum)) + \
       datetime.timedelta(days=days) - datetime.timedelta(days = 366)
    try: v = float( a[1] )
    except: v = numpy.NaN
    flat_tt.append(t)
    flat_vv.append(v)
f.close()

my_whites = mkw.prewhite( numpy.asarray(flat_vv), numpy.asarray(flat_tt), 0.2, alpha_ak=95 )

def test_prewhite( basename ):

    diff = my_whites["pw"] - numpy.loadtxt( basename + ".pw.csv" )
    mse = numpy.nansum(numpy.square(diff)) / len(diff)
    if mse > 1E-3:
        print( "pw mse: " + str(mse) )
    else:
        print( "pw ok" )

    diff = my_whites["pw_cor"] - numpy.loadtxt( basename + ".pw_cor.csv" )
    mse = numpy.nansum(numpy.square(diff)) / len(diff)
    if mse > 1E-3:
        print( "pw_cor mse: " + str(mse) )
    else:
        print( "pw_cor ok" )

    diff = my_whites["tfpw_y"] - numpy.loadtxt( basename + ".tfpw_y.csv" )
    mse = numpy.nansum(numpy.square(diff)) / len(diff)
    if mse > 1E-3:
        print( "tfpw_y mse: " + str(mse) )
    else:
        print( "tfpw_y ok" )

    diff = my_whites["tfpw_ws"] - numpy.loadtxt( basename + ".tfpw_ws.csv" )
    mse = numpy.nansum(numpy.square(diff)) / len(diff)
    if mse > 1E-3:
        print( "tfpw_ws mse: " + str(mse) )
    else:
        print( "tfpw_ws ok" )

    diff = my_whites["vctfpw"] - numpy.loadtxt( basename + ".vctfpw.csv" )
    mse = numpy.nansum(numpy.square(diff)) / len(diff)
    if mse > 1E-3:
        print( "vctfpw mse: " + str(mse) )
    else:
        print( "vctfpw ok" )

    
def test_compute_mk_stat( basename ):
    good_results = numpy.loadtxt( basename + ".results.csv" )

    (result, s, vari, z) = mk.compute_mk_stat( numpy.asarray(flat_tt), my_whites["pw"], 0.2 )
    assert result["p"]    - good_results[0][0] < 1E-10
    assert result["ss"]   - good_results[0][1] == 0
    assert result["slope"]- good_results[0][2] < 1E-10
    assert result["ucl"]  - good_results[0][3] < 1E-15
    assert result["lcl"]  - good_results[0][4] < 1E-15
    assert s    - good_results[0][5] < 1E-2
    assert vari - good_results[0][6] < 1E-10
    assert z    - good_results[0][7] < 1E-10

    (result, s, vari, z) = mk.compute_mk_stat( numpy.asarray(flat_tt), my_whites["pw_cor"], 0.2 )
    assert result["p"]    - good_results[1][0] < 1E-10
    assert result["ss"]   - good_results[1][1] == 0
    assert result["slope"]- good_results[1][2] < 1E-10
    assert result["ucl"]  - good_results[1][3] < 1E-15
    assert result["lcl"]  - good_results[1][4] < 1E-15
    assert s    - good_results[1][5] < 1E-2
    assert vari - good_results[1][6] < 1E-10
    assert z    - good_results[1][7] < 1E-10

    (result, s, vari, z) = mk.compute_mk_stat( numpy.asarray(flat_tt), my_whites["tfpw_y"], 0.2 )
    assert result["p"]    - good_results[2][0] < 1E-10
    assert result["ss"]   - good_results[2][1] == 0
    assert result["slope"]- good_results[2][2] < 1E-10
    assert result["ucl"]  - good_results[2][3] < 1E-15
    assert result["lcl"]  - good_results[2][4] < 1E-15
    assert s    - good_results[2][5] < 1E-2
    assert vari - good_results[2][6] < 1E-10
    assert z    - good_results[2][7] < 1E-10

    (result, s, vari, z) = mk.compute_mk_stat( numpy.asarray(flat_tt), my_whites["tfpw_ws"], 0.2 )
    assert result["p"]    - good_results[3][0] < 1E-10
    assert result["ss"]   - good_results[3][1] == 0
    assert result["slope"]- good_results[3][2] < 1E-10
    assert result["ucl"]  - good_results[3][3] < 1E-15
    assert result["lcl"]  - good_results[3][4] < 1E-15
    assert s    - good_results[3][5] < 1E-2
    assert vari - good_results[3][6] < 1E-10
    assert z    - good_results[3][7] < 1E-10

    (result, s, vari, z) = mk.compute_mk_stat( numpy.asarray(flat_tt), my_whites["vctfpw"], 0.2 )
    assert result["p"]    - good_results[4][0] < 1E-10
    assert result["ss"]   - good_results[4][1] == 0
    assert result["slope"]- good_results[4][2] < 1E-10
    assert result["ucl"]  - good_results[4][3] < 1E-15
    assert result["lcl"]  - good_results[4][4] < 1E-15
    assert s    - good_results[4][5] < 1E-2
    assert vari - good_results[4][6] < 1E-10
    assert z    - good_results[4][7] < 1E-10


test_prewhite( sys.argv[1] )
#test_compute_mk_stat( sys.argv[1] )

