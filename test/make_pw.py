#!/usr/bin/python3

import sys
import numpy
import mannkendall.mk_white as mkw
import mannkendall as mk
import datetime

def mat2datetime(datenum):
    # Matlab time is days since 1/1/0000.
    # Python ordinal time is since 1/1/0001.
    if type(datenum) is int:
        dt = datetime.date.fromordinal(datenum) - \
             datetime.timedelta(days = 366)
        return dt
    else:
        # fraction of the day
        days = datenum % 1
        # add integer days and fraction of the day
        dt = datetime.datetime.fromordinal(int(datenum)) + \
             datetime.timedelta(days=days) - datetime.timedelta(days = 366)
        return dt

d = numpy.loadtxt( sys.argv[1] + ".csv" ).T
dts = numpy.array([mat2datetime(item) for item in d[0]])

w = mkw.prewhite( d[1], dts, 0.02, alpha_ak=95)
# w["pw"] or w["pw_cor"] is Von Storch, 1995
# w["tfpw_y"] is Yue et al., 2002
# w["tfpw_ws"] is Wang & Swail, 2001
# w["vctfpw"] is Wang et al., 2015

numpy.savetxt( "pw", w["pw"] )
numpy.savetxt( "pw_cor", w["pw_cor"] )
numpy.savetxt( "tfpw_y", w["tfpw_y"] )
numpy.savetxt( "tfpw_ws", w["tfpw_ws"] )
numpy.savetxt( "vctfpw", w["vctfpw"] )

for white_name in ["pw","pw_cor","tfpw_y","tfpw_ws","vctfpw"]:
    (result, s, vari, z) = mk.compute_mk_stat( dts, w[white_name], 0.2 )
    print(white_name)
    print((result, s, vari, z))
