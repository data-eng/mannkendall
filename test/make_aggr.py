#!/usr/bin/python3

import sys
import numpy

d = numpy.loadtxt( sys.argv[1] + ".csv" )

acc_3h_dts = numpy.zeros(3)
acc_3h_rfe = numpy.zeros(3)
acc_3h_ssa = numpy.zeros(3)
acc_3h_scattering = numpy.zeros(3)
acc_6h_dts = numpy.zeros(6)
acc_6h_rfe = numpy.zeros(6)
acc_6h_ssa = numpy.zeros(6)
acc_6h_scattering = numpy.zeros(6)
acc_d_dts = numpy.zeros(24)
acc_d_rfe = numpy.zeros(24)
acc_d_ssa = numpy.zeros(24)
acc_d_scattering = numpy.zeros(24)

for i in range(d.shape[0]):
    acc_3h_dts[i%3] = d[i][0]
    acc_3h_rfe[i%3] = d[i][1]
    acc_3h_ssa[i%3] = d[i][2]
    acc_3h_scattering[i%3] = d[i][3]
    acc_6h_dts[i%6] = d[i][0]
    acc_6h_rfe[i%6] = d[i][1]
    acc_6h_ssa[i%6] = d[i][2]
    acc_6h_scattering[i%6] = d[i][3]
    acc_d_dts[i%24] = d[i][0]
    acc_d_rfe[i%24] = d[i][1]
    acc_d_ssa[i%24] = d[i][2]
    acc_d_scattering[i%24] = d[i][3]
    
    print( "rfe1 " + str(d[i][0]) + " " + str(d[i][1]) )
    print( "ssa1 " + str(d[i][0]) + " " + str(d[i][2]) )
    print( "sca1 " + str(d[i][0]) + " " + str(d[i][3]) )
    
    if i % 3 == 2:
        m = numpy.nanmedian( acc_3h_rfe )
        print( "rfe3 " + str(d[i][0]) + " " + str(m) )
        m = numpy.nanmedian( acc_3h_ssa )
        print( "ssa3 " + str(d[i][0]) + " " + str(m) )
        m = numpy.nanmedian( acc_3h_scattering )
        print( "sca3 " + str(d[i][0]) + " " + str(m) )
    if i % 6 == 5:
        m = numpy.nanmedian( acc_6h_rfe )
        print( "rfe6 " + str(d[i][0]) + " " + str(m) )
        m = numpy.nanmedian( acc_6h_ssa )
        print( "ssa6 " + str(d[i][0]) + " " + str(m) )
        m = numpy.nanmedian( acc_6h_scattering )
        print( "sca6 " + str(d[i][0]) + " " + str(m) )
    if i % 24 == 23:
        m = numpy.nanmedian( acc_d_rfe )
        print( "rfed " + str(d[i][0]) + " " + str(m) )
        m = numpy.nanmedian( acc_d_ssa )
        print( "ssad " + str(d[i][0]) + " " + str(m) )
        m = numpy.nanmedian( acc_d_scattering )
        print( "scad " + str(d[i][0]) + " " + str(m) )

