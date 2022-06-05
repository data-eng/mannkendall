#!/home/konstant/bin/python3

import sys
import datetime
import numpy
import mannkendall as mk
import mannkendall.mk_white as mkw

d = numpy.loadtxt( basename + ".csv" ).T
w = mkw.prewhite( d, 0.02, alpha_ak=95)
# w["pw"] or w["pw_cor"] is Von Storch, 1995
# w["tfpw_y"] is Yue et al., 2002
# w["tfpw_ws"] is Wang & Swail, 2001
# w["vctfpw"] is Wang et al., 2015

numpy.savetxt( "pw", w["pw"] )
numpy.savetxt( "pw_cor", w["pw_cor"] )
numpy.savetxt( "tfpw_y", w["tfpw_y"] )
numpy.savetxt( "tfpw_ws", w["tfpw_ws"] )
numpy.savetxt( "vctfpw", w["vctfpw"] )

