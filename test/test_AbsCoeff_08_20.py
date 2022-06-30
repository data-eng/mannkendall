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


def test_prewhite( basename ):
    d = numpy.loadtxt( basename + ".csv" ).T
    my_whites = mkw.prewhite( d, 0.2, alpha_ak=95 )

    try:
        diff = my_whites["pw"] - numpy.loadtxt( basename + ".pw.csv" )
        mse = numpy.nansum(numpy.square(diff)) / len(diff)
        if report:
            print( "pw mse is: " + str(mse) )
        else:
            assert mse < 1E-3
    except FileNotFoundError:
        print("New prewhites. Must record.")

    try:
        diff = my_whites["pw_cor"] - numpy.loadtxt( basename + ".pw_cor.csv" )
        mse = numpy.nansum(numpy.square(diff)) / len(diff)
        if report:
            print( "pw_cor mse is: " + str(mse) )
        else:
            assert mse < 1E-3
    except FileNotFoundError:
        print("New prewhites. Must record.")

    try:
        diff = my_whites["tfpw_y"] - numpy.loadtxt( basename + ".tfpw_y.csv" )
        mse = numpy.nansum(numpy.square(diff)) / len(diff)
        if report:
            print( "tfpw_y mse is: " + str(mse) )
        else:
            assert mse < 1E-3
    except FileNotFoundError:
        mse = None
 
    try:
        diff = my_whites["tfpw_ws"] - numpy.loadtxt( basename + ".tfpw_ws.csv" )
        mse = numpy.nansum(numpy.square(diff)) / len(diff)
        if report:
            print( "tfpw_ws mse is: " + str(mse) )
        else:
            assert mse < 1E-3
    except FileNotFoundError:
        print("New prewhites. Must record.")
        #numpy.savetxt( "strange.tfpw_ws.csv", my_whites["tfpw_ws"] )

    try:
        diff = my_whites["vctfpw"] - numpy.loadtxt( basename + ".vctfpw.csv" )
        mse = numpy.nansum(numpy.square(diff)) / len(diff)
        if report:
            print( "vctfpw mse is: " + str(mse) )
        else:
            assert mse < 1E-3
    except FileNotFoundError:
        print("New prewhites. Must record.")
        #numpy.savetxt( "strange.tfpw_ws.csv", my_whites["tfpw_ws"] )

    return my_whites
    

def test_compute_mk_stat( basename ):
    d = numpy.loadtxt( basename + ".csv" ).T
    #ts = numpy.array( [mkt.mat2datetime(t) for t in d[0]] )
    ts = d[0,:]

    try:
        good_results = numpy.loadtxt( basename + ".results.csv" )
        if report:
            print("good_results")
            print(good_results)
    except:
        good_results = None
    
    if redo_prewhite:
        w = my_whites["pw"]
    else:
        w = numpy.loadtxt( basename + ".pw.csv" )
    dd = numpy.stack( (ts,w), axis=0 )
    (result, s, vari, z) = mk.compute_mk_stat( dd, 0.2 )
    if report:
        print((result, s, vari, z))
    else:
        assert result["p"]    - good_results[0][0] < 1E-8
        assert result["ss"]   - good_results[0][1] == 0
        assert result["slope"]- good_results[0][2] < 1E-8
        assert result["ucl"]  - good_results[0][3] < 1E-8
        assert result["lcl"]  - good_results[0][4] < 1E-8
        assert s    - good_results[0][5] < 1E-2
        assert vari - good_results[0][6] < 1E-8
        assert z    - good_results[0][7] < 1E-8

    if redo_prewhite:
        w = my_whites["pw_cor"]
    else:
        w = numpy.loadtxt( basename + ".pw_cor.csv" )
    dd = numpy.stack( (ts,w), axis=0 )
    (result, s, vari, z) = mk.compute_mk_stat( dd, 0.2 )
    if report:
        print((result, s, vari, z))
    else:
        assert result["p"]    - good_results[1][0] < 1E-8
        assert result["ss"]   - good_results[1][1] == 0
        assert result["slope"]- good_results[1][2] < 1E-8
        assert result["ucl"]  - good_results[1][3] < 1E-8
        assert result["lcl"]  - good_results[1][4] < 1E-8
        assert s    - good_results[1][5] < 1E-2
        assert vari - good_results[1][6] < 1E-8
        assert z    - good_results[1][7] < 1E-8

    if redo_prewhite:
        w = my_whites["tfpw_y"]
    else:
        w = numpy.loadtxt( basename + ".tfpw_y.csv" )
    dd = numpy.stack( (ts,w), axis=0 )
    (result, s, vari, z) = mk.compute_mk_stat( dd, 0.2 )
    if report:
        print((result, s, vari, z))
    else:
        assert result["p"]    - good_results[2][0] < 1E-8
        assert result["ss"]   - good_results[2][1] == 0
        assert result["slope"]- good_results[2][2] < 1E-8
        assert result["ucl"]  - good_results[2][3] < 1E-8
        assert result["lcl"]  - good_results[2][4] < 1E-8
        assert s    - good_results[2][5] < 1E-2
        assert vari - good_results[2][6] < 1E-8
        assert z    - good_results[2][7] < 1E-8

    if redo_prewhite:
        w = my_whites["tfpw_ws"]
    else:
        w = numpy.loadtxt( basename + ".tfpw_ws.csv" )
    dd = numpy.stack( (ts,w), axis=0 )
    (result, s, vari, z) = mk.compute_mk_stat( dd, 0.2 )
    if report:
        print((result, s, vari, z))
    else:
        assert result["p"]    - good_results[3][0] < 1E-8
        assert result["ss"]   - good_results[3][1] == 0
        assert result["slope"]- good_results[3][2] < 1E-8
        assert result["ucl"]  - good_results[3][3] < 1E-8
        assert result["lcl"]  - good_results[3][4] < 1E-8
        assert s    - good_results[3][5] < 1E-2
        assert vari - good_results[3][6] < 1E-8
        assert z    - good_results[3][7] < 1E-8

    if redo_prewhite:
        w = my_whites["vctfpw"]
    else:
        w = numpy.loadtxt( basename + ".vctfpw.csv" )
    dd = numpy.stack( (ts,w), axis=0 )
    (result, s, vari, z) = mk.compute_mk_stat( dd, 0.2 )
    if report:
        print((result, s, vari, z))
    else:
        assert result["p"]    - good_results[4][0] < 1E-8
        assert result["ss"]   - good_results[4][1] == 0
        assert result["slope"]- good_results[4][2] < 1E-8
        assert result["ucl"]  - good_results[4][3] < 1E-8
        assert result["lcl"]  - good_results[4][4] < 1E-8
        assert s    - good_results[4][5] < 1E-2
        assert vari - good_results[4][6] < 1E-8
        assert z    - good_results[4][7] < 1E-8


report = False
redo_prewhite = True


if redo_prewhite:
    my_whites = test_prewhite( sys.argv[1] )
else:
    my_whites = None
    
test_compute_mk_stat( sys.argv[1] )

