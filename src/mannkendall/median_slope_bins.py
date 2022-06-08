import sys
import numpy
import time
import operator


def initializer( obs ):
    retv = {}
    retv["obs"] = obs
    # Three array of up-to-this size will be needed
    # for the second pass
    retv["max_size"] = 1048576
    retv["trace"] = True
    # These are the three arrays
    retv["lo"] = None
    retv["me"] = None
    retv["hi"] = None

    (_,l) = obs.shape
    min = None
    max = None
    for i in range(1,l):
        slope = float(obs[1][i]-obs[1][i-1]) / float(obs[0][i]-obs[0][i-1])
        try:
            if slope < min: min = slope
        except TypeError:
            min = slope
        try:
            if slope > max: max = slope
        except TypeError:
            max = slope

    n = (l-2)*(l-1)/2
    retv["num_bins"] = 2 * int(1 + n // retv["max_size"])
    bin_width = (max-min)/retv["num_bins"]

    if retv["trace"]:
        print( "Init: " + str(retv["num_bins"]) + " bins for " + str(n) + " items from " + str(min) + " to " + str(max)  )
        print( "bin_width is " + str(bin_width) )
    q = []
    b = min+bin_width
    while( b < max ):
        q.append( b )
        b += bin_width
    retv["bin_boundary"] = numpy.array( q )
    retv["bin_count"] = numpy.array( [0] * retv["num_bins"] )
    if retv["trace"]:
        print( retv["bin_boundary"] )
        print( retv["bin_count"] )
    
    return retv



def record_value( d, v, f ):
    # f is 1 or 0.
    # Use 1 to increment the bin count.
    # Use 0 to only find the righ bin
    bin=0
    while( (bin < d["num_bins"]-1) and (v > d["bin_boundary"][bin]) ):
        bin += 1
    if d["trace"] and False:
        print( "Added " + str(v) + " to bin " + str(bin) )
    d["bin_count"][bin] += f
    return bin


    
def rebalance( d ):
    max_idx = d["bin_count"].argmax()
    if d["bin_count"][max_idx] > d["max_size"]:
        # A bin has gotten too big. Split in half and
        # merge the smallest bin with its neighbour
        # to maintain the number of bins.

        min_idx = d["bin_count"].argmin()
        if min_idx == 0:
            bin1 = 0
            bin2 = 1
        elif min_idx == d["num_bins"] - 1:
            bin1 = d["num_bins"] - 2
            bin2 = d["num_bins"] - 1
        elif d["bin_count"][min_idx-1] > d["bin_count"][min_idx+1]:
            bin1 = min_idx
            bin2 = min_idx + 1
        else:
            bin1 = min_idx - 1
            bin2 = min_idx

        if bin2 < max_idx:
            d["bin_count"][bin1] += d["bin_count"][bin2]
            d["bin_boundary"][bin1] = d["bin_boundary"][bin2]
            for i in range(bin2,max_idx-1):
                d["bin_count"][i] = d["bin_count"][i+1]
                d["bin_boundary"][i] = d["bin_boundary"][i+1]
            try:
                d["bin_boundary"][max_idx-1] = (d["bin_boundary"][max_idx]+d["bin_boundary"][max_idx-1]) / 2
            except:
                print("XX")
                print(max_idx)
                print(min_idx)
                d["bin_boundary"][max_idx-1] = (d["bin_boundary"][max_idx]+d["bin_boundary"][max_idx-1]) / 2
                
            d["bin_count"][max_idx-1] = d["bin_count"][max_idx] // 2
            if d["bin_count"][max_idx] % 2 == 0:
                d["bin_count"][max_idx] = d["bin_count"][max_idx] // 2
            else:
                d["bin_count"][max_idx] = 1 + d["bin_count"][max_idx] // 2
        else:
            d["bin_count"][bin2] += d["bin_count"][bin1]
            for i in range(bin1-1,max_idx,-1):
                d["bin_count"][i+1] = d["bin_count"][i]
                d["bin_boundary"][i+1] = d["bin_boundary"][i]
            d["bin_boundary"][max_idx+1] = d["bin_boundary"][max_idx]
            d["bin_boundary"][max_idx] = (d["bin_boundary"][max_idx]+d["bin_boundary"][max_idx-1]) / 2
            d["bin_count"][max_idx+1] = d["bin_count"][max_idx] // 2
            if d["bin_count"][max_idx] % 2 == 0:
                d["bin_count"][max_idx] = d["bin_count"][max_idx] // 2
            else:
                d["bin_count"][max_idx] = 1 + d["bin_count"][max_idx] // 2
        return True
    else:
        return False



def median( d ):
    n = d["left_heap"] + d["right_heap"] + len(d["data"])
    if d["trace"]:
        print("MM1 " + str(n) )
    if n == 0:
        retv = None
    elif n == 1:
        (_,_,retv) = d["data"][0]
    elif n%2 == 0:
        idx_median = n//2 - d["left_heap"]
        if d["trace"]:
            print("MM2a " + str(idx_median) )
        if (idx_median < 0) or (idx_median >= len(d["data"])):
            # we lost the median somewhere in the heaps
            retv = None
        else:
            (_,_,retv) = d["data"][idx_median]
    else:
        idx_median = n//2 - d["left_heap"]
        if d["trace"]:
            print("MM2b " + str(idx_median) )
        if (idx_median < 0) or (idx_median >= len(d["data"])):
            # we lost the median somewhere in the heaps
            retv = None
        else:
            (_,_,m1) = d["data"][idx_median]
            (_,_,m2) = d["data"][idx_median+1]
            retv = float(m1+m2)/2
    return retv



def find_bins( d, low=0.05, med=0.5, high=0.95 ):
    (_,l) = d["obs"].shape
    print((l-2)*(l-1)/2)
    n=0
    for i in range(1,l):
        for j in range(i+1,l):
            n += 1
            slope = float(obs[1][j]-obs[1][i]) / float(obs[0][j]-obs[0][i])
            record_value( d, slope, 1 )
            if n%10000 == 0:
                if d["trace"] and False:
                    print( str(n)+": "+str(d["bin_count"]) )
                r = rebalance( d )
                if d["trace"] and r:
                    print("Rebalanced")
                    print( d["bin_count"] )
                    print( d["bin_boundary"] )
    if d["trace"]:
        print("The end")
        print(n)
        print( d["bin_count"] )
        print( d["bin_boundary"] )

    percentile05 = 0.05 * float(n)
    percentile50 = 0.5  * float(n)
    percentile95 = 0.95 * float(n)

    percentile05_bin = None
    percentile50_bin = None
    percentile95_bin = None

    acc = 0
    for b in range(d["num_bins"]):
        acc += d["bin_count"][b]
        if (percentile05_bin is None) and (percentile05 < acc):
            percentile05_bin = b
        if (percentile50_bin is None) and (percentile50 < acc):
            percentile50_bin = b
        if (percentile95_bin is None) and (percentile95 < acc):
            percentile95_bin = b

    return( percentile05_bin, percentile50_bin, percentile95_bin )



def populate_bins( d, low, med, high ):
    (_,l) = d["obs"].shape
    d["lo"] = numpy.zeros( d["max_size"] )
    d["me"] = numpy.zeros( d["max_size"] )
    d["hi"] = numpy.zeros( d["max_size"] )
    if d["trace"]:
        print( 'd["lo"] has '+str(len(d["lo"])) )
        print( 'd["me"] has '+str(len(d["me"])) )
        print( 'd["hi"] has '+str(len(d["hi"])) )
    lo_ptr = 0
    me_ptr = 0
    hi_ptr = 0
    n=0
    for i in range(1,l):
        for j in range(i+1,l):
            n+=1
            slope = float(obs[1][j]-obs[1][i]) / float(obs[0][j]-obs[0][i])
            bin = record_value( d, slope, -1 )
            if d["trace"] and ( (lo_ptr > 1048570) or (me_ptr > 1048570) or (hi_ptr > 1048570) ):
                print( n )
                print( lo_ptr )
                print( bin )
                print( d["bin_boundary"] )
                print( d["bin_count"] )
            if bin == low:
                d["lo"][lo_ptr] = slope
                lo_ptr += 1
            if bin == med:
                d["me"][me_ptr] = slope
                me_ptr += 1
            if bin == high:
                d["hi"][hi_ptr] = slope
                hi_ptr += 1

    print( "Polulated bins with: low ("+str(d["bin_count"][low])+"): "+str(lo_ptr)+\
           " med ("+str(d["bin_count"][med])+"): "+str(me_ptr)++
           " high ("+str(d["bin_count"][high])+")"+str(hi_ptr) )



test = "test2"

if test == "test1":
    data = [ [0,10,20,30,40,50,60,70,80,90,100,110,120], [5,1,5,6,5,2,4,8,2,3,1,2,4] ]
    obs = numpy.array(data)
    #a=datastack[:,datastack[1,:].argsort()]
    #print(datastack)
else:
    obs1 = numpy.loadtxt( sys.argv[1] ).T
    print( "loaded " + str(obs1.shape) )
    obsT = obs1.T
    obs = ((obsT)[~numpy.isnan(obsT).any(axis=1)]).T
    print( "after nan rm " + str(obs.shape) )

d = initializer( obs )
(low_bin, mid_bin, high_bin) = find_bins( d )
populate_bins( d, low_bin, mid_bin, high_bin )
