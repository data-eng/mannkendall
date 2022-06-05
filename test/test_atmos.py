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
    my_whites = mkw.prewhite( d, 0.02, alpha_ak=95 )

    diff = my_whites["pw"] - numpy.loadtxt( basename + ".pw.csv" )
    mse = numpy.nansum(numpy.square(diff)) / len(diff)
    print( "pw: "+str(mse)+" "+str(numpy.nanmean(my_whites["pw"])) )
    numpy.savetxt( basename+".pw", my_whites["pw"] )
    #assert numpy.nansum(numpy.square(diff)) < 1E-10

    diff = my_whites["pw_cor"] - numpy.loadtxt( basename + ".pw_cor.csv" )
    mse = numpy.nansum(numpy.square(diff)) / len(diff)
    print( "pw_corr: "+str(mse)+" "+str(numpy.nanmean(my_whites["pw_cor"])) )
    numpy.savetxt( basename+".pw_cor", my_whites["pw_cor"] )
    #assert numpy.nansum(numpy.square(diff)) < 1E-10

    diff = my_whites["tfpw_y"] - numpy.loadtxt( basename + ".tfpw_y.csv" )
    mse = numpy.nansum(numpy.square(diff)) / len(diff)
    print( "tfpw_y: "+str(mse)+" "+str(numpy.nanmean(my_whites["tfpw_y"])) )
    numpy.savetxt( basename+".tfpw_y", my_whites["tfpw_y"] )
    #assert numpy.nansum(numpy.square(diff)) < 1E-10

    diff = my_whites["tfpw_ws"] - numpy.loadtxt( basename + ".tfpw_ws.csv" )
    mse = numpy.nansum(numpy.square(diff)) / len(diff)
    print( "tfpw_ws: "+str(mse)+" "+str(numpy.nanmean(my_whites["tfpw_ws"])) )
    numpy.savetxt( basename+".tfpw_ws", my_whites["tfpw_ws"] )
    #assert numpy.nansum(numpy.square(diff)) < 1E-10

    diff = my_whites["vctfpw"] - numpy.loadtxt( basename + ".vctfpw.csv" )
    mse = numpy.nansum(numpy.square(diff)) / len(diff)
    print( "vctfpw: "+str(mse)+" "+str(numpy.nanmean(my_whites["vctfpw"])) )
    numpy.savetxt( basename+".vctfpw", my_whites["vctfpw"] )
    #assert numpy.nansum(numpy.square(diff)) < 1E-10

    
def test_compute_mk_stat( basename ):
    d = numpy.loadtxt( basename + ".csv" ).T
    good_results = numpy.loadtxt( basename + ".results.csv" )

    w = numpy.loadtxt( basename + ".pw.csv" )
    dd = numpy.stack( (d[0,:],w), axis=0 )
        
    (result, s, vari, z) = mk.compute_mk_stat( dd, 0.02 )
    assert result["p"]    - good_results[0][0] < 1E-10
    assert result["ss"]   - good_results[0][1] == 0
    assert result["slope"]- good_results[0][2] < 1E-10
    #assert result["ucl"]  - good_results[0][3] < 1E-15
    #assert result["lcl"]  - good_results[0][4] < 1E-15
    assert s    - good_results[0][5] < 1E-2
    assert vari - good_results[0][6] < 1E-10
    assert z    - good_results[0][7] < 1E-10

    w = numpy.loadtxt( basename + ".pw_cor.csv" )
    dd = numpy.stack( (d[0,:],w), axis=0 )
    (result, s, vari, z) = mk.compute_mk_stat( dd, 0.02 )
    assert result["p"]    - good_results[1][0] < 1E-10
    assert result["ss"]   - good_results[1][1] == 0
    assert result["slope"]- good_results[1][2] < 1E-10
    #assert result["ucl"]  - good_results[1][3] < 1E-15
    #assert result["lcl"]  - good_results[1][4] < 1E-15
    assert s    - good_results[1][5] < 1E-2
    assert vari - good_results[1][6] < 1E-10
    assert z    - good_results[1][7] < 1E-10

    w = numpy.loadtxt( basename + ".tfpw_y.csv" )
    dd = numpy.stack( (d[0,:],w), axis=0 )
    (result, s, vari, z) = mk.compute_mk_stat( dd, 0.02 )
    assert result["p"]    - good_results[2][0] < 1E-10
    assert result["ss"]   - good_results[2][1] == 0
    assert result["slope"]- good_results[2][2] < 1E-10
    #assert result["ucl"]  - good_results[2][3] < 1E-15
    #assert result["lcl"]  - good_results[2][4] < 1E-15
    assert s    - good_results[2][5] < 1E-2
    assert vari - good_results[2][6] < 1E-10
    assert z    - good_results[2][7] < 1E-10

    w = numpy.loadtxt( basename + ".tfpw_ws.csv" )
    dd = numpy.stack( (d[0,:],w), axis=0 )
    (result, s, vari, z) = mk.compute_mk_stat( dd, 0.02 )
    assert result["p"]    - good_results[3][0] < 1E-10
    assert result["ss"]   - good_results[3][1] == 0
    assert result["slope"]- good_results[3][2] < 1E-10
    #assert result["ucl"]  - good_results[3][3] < 1E-15
    #assert result["lcl"]  - good_results[3][4] < 1E-15
    assert s    - good_results[3][5] < 1E-2
    assert vari - good_results[3][6] < 1E-10
    assert z    - good_results[3][7] < 1E-10

    w = numpy.loadtxt( basename + ".vctfpw.csv" )
    dd = numpy.stack( (d[0,:],w), axis=0 )
    (result, s, vari, z) = mk.compute_mk_stat( dd, 0.02 )
    assert result["p"]    - good_results[4][0] < 1E-10
    assert result["ss"]   - good_results[4][1] == 0
    assert result["slope"]- good_results[4][2] < 1E-10
    #assert result["ucl"]  - good_results[4][3] < 1E-15
    #assert result["lcl"]  - good_results[4][4] < 1E-15
    assert s    - good_results[4][5] < 1E-2
    assert vari - good_results[4][6] < 1E-10
    assert z    - good_results[4][7] < 1E-10


if len(sys.argv) > 1:
    test_prewhite( sys.argv[1] )
    #test_compute_mk_stat( sys.argv[1] )
else:
    test_prewhite( "test_data/AbsCoeff_08_20_daily" )
    test_compute_mk_stat( "test_data/AbsCoeff_08_20_daily" )

