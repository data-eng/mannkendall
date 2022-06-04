# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 NCSR "Demokritos"

Distributed under the terms of the BSD 3-Clause License.

SPDX-License-Identifier: BSD-3-Clause

This file contains test functions
"""

import sys
import numpy

import mannkendall as mk
from mannkendall import mk_stats as mks
from mannkendall import mk_white as mkw
from mannkendall import mk_tools as mkt

tt = []
vv = []

def load_data( datafile ):
    f = open( datafile, "r" )
    while True:
        line = f.readline()
        if not line: break
        a = line.split( "," )
        t = float( a[0] )
        try: v = float( a[1] )
        except: v = numpy.NaN
        tt.append(t)
        vv.append(v)
    f.close()
    return numpy.array( [tt, vv], float )


def test_compute_mk_stat( basename ):
    d = load_data( basename + ".csv" )
    tt = []
    for t in d[0,:]:
        tt.append( mkt.mat2datetime(t) ) 
    good_results = numpy.loadtxt( basename + ".results.csv" )

    w = numpy.loadtxt( basename + ".pw.csv" )
    dd = numpy.stack( (d[0,:],w), axis=0 )
    (result, s, vari, z) = mk.compute_mk_stat( dd, 0.02 )
    assert result["p"]    - good_results[0][0] < 1E-10
    assert result["ss"]   - good_results[0][1] == 0
    assert result["slope"]- good_results[0][2] < 1E-10
    assert result["ucl"]  - good_results[0][3] < 1E-10
    assert result["lcl"]  - good_results[0][4] < 1E-10
    assert s    - good_results[0][5] < 1E-2
    assert vari - good_results[0][6] < 1E-10
    assert z    - good_results[0][7] < 1E-10

    w = numpy.loadtxt( basename + ".pw_cor.csv" )
    dd = numpy.stack( (d[0,:],w), axis=0 )
    (result, s, vari, z) = mk.compute_mk_stat( dd, 0.02 )
    assert result["p"]    - good_results[1][0] < 1E-10
    assert result["ss"]   - good_results[1][1] == 0
    assert result["slope"]- good_results[1][2] < 1E-10
    assert result["ucl"]  - good_results[1][3] < 1E-10
    assert result["lcl"]  - good_results[1][4] < 1E-10
    assert s    - good_results[1][5] < 1E-2
    assert vari - good_results[1][6] < 1E-10
    assert z    - good_results[1][7] < 1E-10

    w = numpy.loadtxt( basename + ".tfpw_y.csv" )
    dd = numpy.stack( (d[0,:],w), axis=0 )
    (result, s, vari, z) = mk.compute_mk_stat( dd, 0.02 )
    assert result["p"]    - good_results[2][0] < 1E-10
    assert result["ss"]   - good_results[2][1] == 0
    assert result["slope"]- good_results[2][2] < 1E-10
    assert result["ucl"]  - good_results[2][3] < 1E-10
    assert result["lcl"]  - good_results[2][4] < 1E-10
    assert s    - good_results[2][5] < 1E-2
    assert vari - good_results[2][6] < 1E-10
    assert z    - good_results[2][7] < 1E-10

    w = numpy.loadtxt( basename + ".tfpw_ws.csv" )
    dd = numpy.stack( (d[0,:],w), axis=0 )
    (result, s, vari, z) = mk.compute_mk_stat( dd, 0.02 )
    assert result["p"]    - good_results[3][0] < 1E-10
    assert result["ss"]   - good_results[3][1] == 0
    assert result["slope"]- good_results[3][2] < 1E-10
    assert result["ucl"]  - good_results[3][3] < 1E-10
    assert result["lcl"]  - good_results[3][4] < 1E-10
    assert s    - good_results[3][5] < 1E-2
    assert vari - good_results[3][6] < 1E-10
    assert z    - good_results[3][7] < 1E-10

    w = numpy.loadtxt( basename + ".vctfpw.csv" )
    dd = numpy.stack( (d[0,:],w), axis=0 )
    (result, s, vari, z) = mk.compute_mk_stat( dd, 0.02 )
    assert result["p"]    - good_results[4][0] < 1E-10
    assert result["ss"]   - good_results[4][1] == 0
    assert result["slope"]- good_results[4][2] < 1E-10
    assert result["ucl"]  - good_results[4][3] < 1E-10
    assert result["lcl"]  - good_results[4][4] < 1E-10
    assert s    - good_results[4][5] < 1E-2
    assert vari - good_results[4][6] < 1E-10
    assert z    - good_results[4][7] < 1E-10


test_compute_mk_stat( sys.argv[1] )

