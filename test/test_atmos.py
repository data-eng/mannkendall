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

    return

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

    #res = mk.compute_mk_stat( numpy.array(tt), w["tfpw_y"], 0.02 )
    #res = mk.compute_mk_stat( numpy.array(tt), w["tfpw_ws"], 0.02 )
    #res = mk.compute_mk_stat( numpy.array(tt), w["vctfpw"], 0.02 )


test_compute_mk_stat( sys.argv[1] )

