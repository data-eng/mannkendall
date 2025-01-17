# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 NCSR "Demokritos"

Distributed under the terms of the BSD 3-Clause License.

SPDX-License-Identifier: BSD-3-Clause

This file contains test functions
"""

import sys
import numpy
import json

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
        numpy.savetxt( basename + ".pw.csv", my_whites["pw"] )

    try:
        diff = my_whites["pw_cor"] - numpy.loadtxt( basename + ".pw_cor.csv" )
        mse = numpy.nansum(numpy.square(diff)) / len(diff)
        if report:
            print( "pw_cor mse is: " + str(mse) )
        else:
            assert mse < 1E-3
    except FileNotFoundError:
        print("New prewhites. Must record.")
        numpy.savetxt( basename + ".pw_cor.csv", my_whites["pw_cor"] )

    try:
        diff = my_whites["tfpw_y"] - numpy.loadtxt( basename + ".tfpw_y.csv" )
        mse = numpy.nansum(numpy.square(diff)) / len(diff)
        if report:
            print( "tfpw_y mse is: " + str(mse) )
        else:
            assert mse < 1E-3
    except FileNotFoundError:
        print("New prewhites. Must record.")
        numpy.savetxt( basename + ".tfpw_y.csv", my_whites["tfpw_y"] )
 
    try:
        diff = my_whites["tfpw_ws"] - numpy.loadtxt( basename + ".tfpw_ws.csv" )
        mse = numpy.nansum(numpy.square(diff)) / len(diff)
        if report:
            print( "tfpw_ws mse is: " + str(mse) )
        else:
            assert mse < 1E-3
    except FileNotFoundError:
        print("New prewhites. Must record.")
        numpy.savetxt( basename + ".tfpw_ws.csv", my_whites["tfpw_ws"] )

    try:
        diff = my_whites["vctfpw"] - numpy.loadtxt( basename + ".vctfpw.csv" )
        mse = numpy.nansum(numpy.square(diff)) / len(diff)
        if report:
            print( "vctfpw mse is: " + str(mse) )
        else:
            assert mse < 1E-3
    except FileNotFoundError:
        print("New prewhites. Must record.")
        numpy.savetxt( basename + ".vctfpw.csv", my_whites["vctfpw"] )

    return my_whites
    

def test_compute_mk_stat( basename ):
    d = numpy.loadtxt( basename + ".csv" ).T
    #ts = numpy.array( [mkt.mat2datetime(t) for t in d[0]] )
    ts = d[0,:]

    try:
        with open( basename + ".results.json", "r" ) as fp:
            good_results = json.load( fp )
        if report:
            print("Saved results are:")
            print(good_results)
    except:
        good_results = None
        new_results = {}


    for white_name in ["pw","pw_cor","tfpw_y","tfpw_ws","vctfpw"]:
        print("Checking MK over "+white_name)
        filename = basename + "." + white_name + ".csv"
        if redo_prewhite:
            w = my_whites[white_name]
        else:
            w = numpy.loadtxt( filename )
        dd = numpy.stack( (ts,w), axis=0 )
        (result, s, vari, z) = mk.compute_mk_stat( dd, 0.2 )

        if good_results is None:
            new_results[white_name] = {}
            new_results[white_name]["p"] = result["p"]
            new_results[white_name]["ss"] = result["ss"]
            new_results[white_name]["slope"] = result["slope"]
            new_results[white_name]["ucl"] = result["ucl"]
            new_results[white_name]["lcl"] = result["lcl"]
            new_results[white_name]["s"] = s
            new_results[white_name]["vari"] = vari
            new_results[white_name]["z"] = z
            with open( basename + ".results.json", "w" ) as fp:
                json.dump( new_results, fp, indent=2 )
        else:
            dp  = result["p"]    - good_results[white_name]["p"]
            dss = result["ss"]   - good_results[white_name]["ss"]
            dsl = result["slope"]- good_results[white_name]["slope"]
            ducl= result["ucl"]  - good_results[white_name]["ucl"]
            dlcl= result["lcl"]  - good_results[white_name]["lcl"]
            ds  = s    - good_results[white_name]["s"]
            dvar= vari - good_results[white_name]["vari"]
            dz  = z    - good_results[white_name]["z"]

            if report:
                print((result, s, vari, z))
            if report:
                print("p error: "+str(dp)+", rel err: "+str(dp/good_results[white_name]["p"]))
            else:
                assert dp   < 1E-8
            if report:
                # Nor rel error, as this is often zero
                print("ss error: "+str(dss))
            else:
                assert dss == 0
            if report:
                print("slope error: "+str(dsl)+", rel err: "+str(dsl/good_results[white_name]["slope"]))
            else:
                assert dsl  < 1E-8
            if report:
                print("ucl error: "+str(ducl)+", rel err: "+str(ducl/good_results[white_name]["ucl"]))
            else:
                assert ducl < 1E-8
            if report:
                print("lcl error: "+str(dlcl)+", rel err: "+str(dlcl/good_results[white_name]["lcl"]))
            else:
                assert dlcl < 1E-8
            if report:
                print("s error: "+str(ds)+", rel err: "+str(ds/good_results[white_name]["s"]))
            else:
                assert ds   < 1E-2
            if report:
                print("vari error: "+str(dvar)+", rel err: "+str(dvar/good_results[white_name]["vari"]))
            else:
                assert dvar < 1E-8
            if report:
                print("z error: "+str(dz)+", rel err: "+str(dz/good_results[white_name]["z"]))
            else:
                assert dz   < 1E-8


report = True
redo_prewhite = True


if redo_prewhite:
    my_whites = test_prewhite( sys.argv[1] )
else:
    my_whites = None
    
test_compute_mk_stat( sys.argv[1] )

