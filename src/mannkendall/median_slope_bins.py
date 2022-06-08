import sys
import numpy
import time
import operator

30501853
8667366

def initializer( obs ):
    retv = {}
    # Three float arrays of this size will be needed
    # for the second pass
    retv["max_size"] = 2048576
    retv["trace"] = True

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



def record_value( d, v ):
    bin=0
    while( (bin < d["num_bins"]-1) and (v > d["bin_boundary"][bin]) ):
        bin += 1
    if d["trace"] and False:
        print( "Added " + str(v) + " to bin " + str(bin) )
    d["bin_count"][bin] += 1


    
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
print( d["bin_count"] )
print( d["bin_boundary"] )
    
all_slopes = []

(_,l) = obs.shape
print((l-2)*(l-1)/2)
n=0
for i in range(1,l):
    for j in range(i+1,l):
        n += 1
        slope = float(obs[1][j]-obs[1][i]) / float(obs[0][j]-obs[0][i])
        #all_slopes.append( (i,j,slope) )
        record_value( d, slope )
        if n%10000 == 0:
            print(n)
            print( d["bin_count"] )
            r = rebalance( d )
            if r:
                print("Rebalanced")
                print( d["bin_count"] )
                print( d["bin_boundary"] )

print("The end")
print(n)
print( d["bin_count"] )
print( d["bin_boundary"] )

#print( median(d) )

    #d.append( all_sorted[0] )
    #d.append( all_sorted[-1] )


