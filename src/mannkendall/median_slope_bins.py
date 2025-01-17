import sys
import numpy
import time
import operator


def initializer( obs, med_idx_1, med_idx_2, lcl_idx=None, ucl_idx=None ):
    retv = {}
    retv["obs"] = obs
    # Three arrays of up-to-this size will be needed
    # for the second pass
    retv["max_size"] = 32000000
    retv["trace"] = False
    # The indexes of the lcl and ucl points
    # If None, lcl and ucle will not be returned.
    retv["lcl_idx"] = lcl_idx
    retv["ucl_idx"] = ucl_idx
    retv["med_idx_1"] = med_idx_1
    retv["med_idx_2"] = med_idx_2
    # These are the three arrays
    retv["lo"] = None
    retv["me"] = None
    retv["hi"] = None

    (_,l) = obs.shape
    # The num of slopes is Sum l-1, l-2, ... 1 = l(l-1)/2
    n = l*(l-1)/2
    retv["n"] = n

    retv["num_bins"] = 2 * int(1 + n // retv["max_size"])
    # it is best to have odd nbins, just in case we are too successful
    # in balancing them and calculating the needs values from different bins
    if retv["num_bins"]%2 == 0:
        retv["num_bins"] += 1
    retv["bin_boundary"] = []

    # We need the distribution of the slopes to set good
    # bin boundaries. We will sample the data to estimate it.
    slope_sample=[]
    sampler = 1
    if l > 2048:
        sampler = 2
    if l > 8192:
        sampler = 16
    if l > 32768:
        sampler = 16384
    if retv["trace"]:
        print("For "+str(l)+" observations, sampling one out of "+str(sampler))
    for i in range(1,l):
        if i%sampler == 0:
            for j in range(i+1,l):
                assert not numpy.isnan(obs[1][i])
                assert not numpy.isnan(obs[1][j])
                slope = (obs[1][j]-obs[1][i]) / float(obs[0][j]-obs[0][i])
                slope_sample.append( slope )

    slope_sample.sort()
    step = len(slope_sample) / retv["num_bins"]
    idx_f = step
    while idx_f < len(slope_sample):
        retv["bin_boundary"].append( slope_sample[int(idx_f)] )
        idx_f += step
    # this can happen depending on how len(slope_sample)/retv["num_bin"]
    # gets rounded into a float. Just drop the right-most value.
    if len(retv["bin_boundary"]) == retv["num_bins"]:
        retv["bin_boundary"].pop()
    assert len(retv["bin_boundary"]) == retv["num_bins"]-1

    # We partitioned into bins that have `step` items each, so we
    # expect the actual bins to have `step` * `sampler` items each.
    expected_bin_size = step * sampler

    # we have not actually accounted, so the contents of bin_count
    # are estimates
    retv["bin_count_accurate"] = False
    retv["bin_count"] = numpy.array( [expected_bin_size] * retv["num_bins"] )
    if retv["trace"]:
        print("Init:")
        print("Bin counts: " + str(retv["bin_count"]) )
        print("Bin boundaries: " + str(retv["bin_boundary"]) )

    # Based on this estimated distribution, also estimate which bins
    # will contain median, lcl, ucl. This allows us to provisionally
    # populate them during the first count, so that we might get our
    # result with only one count.

    if lcl_idx is None: retv["lcl_bin"] = None
    else: retv["lcl_bin"] = int(lcl_idx // expected_bin_size)

    if ucl_idx is None: retv["ucl_bin"] = None
    else: retv["ucl_bin"] = int(ucl_idx // expected_bin_size)

    assert (med_idx_1 // expected_bin_size) == (med_idx_2 // expected_bin_size)
    retv["med_bin"] = int(med_idx_1 // expected_bin_size)

    if retv["trace"]:
        print("Estimated that median (idx: "+str(med_idx_1)+","+str(med_idx_1)+") will be in " + str(retv["med_bin"]) + ", lcl (idx: "+str(lcl_idx)+") in " + str(retv["lcl_bin"]) + ", ucl (idx: "+str(ucl_idx)+") in " + str(retv["ucl_bin"]))

    return retv



def record_value( d, v, f ):
    # f is 1 or 0.
    # Use 1 to increment the bin count.
    # Use 0 to only find the righ bin
    # if pop is True, the med,lcl,ucl bins are also populated
    bin=0
    while( (bin < d["num_bins"]-1) and (v > d["bin_boundary"][bin]) ):
        bin += 1
    if d["trace"] and False:
        print( "Placed " + str(v) + " in bin " + str(bin) )
    d["bin_count"][bin] += f
        
    return bin


    
def rebalance( d ):
    max_idx = d["bin_count"].argmax()
    if d["bin_count"][max_idx] > d["max_size"]:
        # A bin has gotten too big. Split in half and
        # merge the smallest bin with its neighbour
        # to maintain the number of bins.

        min_idx = d["bin_count"].argmin()
        # if the smallest bin is at an edge, merge with the
        # only neighbour it has
        if min_idx == 0:
            bin1 = 0
            bin2 = 1
        elif min_idx == d["num_bins"] - 1:
            bin1 = d["num_bins"] - 2
            bin2 = d["num_bins"] - 1
        # else, select the smallest neighbour
        elif d["bin_count"][min_idx-1] > d["bin_count"][min_idx+1]:
            bin1 = min_idx
            bin2 = min_idx + 1
        else:
            bin1 = min_idx - 1
            bin2 = min_idx


        # If max_idx and min_idx are neighbours, rebalance between them.
        # If this case is not treated specifically and min_idx is at the edge,
        # then this line below is an index-out-of-bound exception:
            #  d["bin_boundary"][max_idx+1] = d["bin_boundary"][max_idx]
        # as there are only max_idx-1 boundaries, so
        # d["bin_boundary"][max_idx+1] is out of range.
        if (bin1==max_idx) or (bin2==max_idx):
            if bin2 == len(d["bin_boundary"]):
                # The right-most bin has no boundary, and must
                # be treated specially.
                # Let us assume the very first bin_width as a decent
                # estimate of bin_width here as well
                w =  d["bin_boundary"][1] - d["bin_boundary"][0]
                d["bin_boundary"][bin1] = d["bin_boundary"][bin1-1] + w
            elif bin1 == 0:
                # The left-most bin has no boundary, and must
                # be treated specially.
                w =  d["bin_boundary"][1] - d["bin_boundary"][0]
                d["bin_boundary"][bin1] = d["bin_boundary"][bin2] - w
            else:
                # The unmarked case: move the first boundary to the middle
                d["bin_boundary"][bin1] = (d["bin_boundary"][bin2]+d["bin_boundary"][bin-1]) / 2
            sum = d["bin_count"][bin1] + d["bin_count"][bin2]
            if sum % 2 == 0:
                d["bin_count"][bin1] = sum // 2
                d["bin_count"][bin2] = sum // 2
            else:
                d["bin_count"][bin1] = sum // 2
                d["bin_count"][bin2] = 1 + sum // 2
            
        
        # the merged bins are to the left of max_idx
        elif bin2 < max_idx:
            d["bin_count"][bin1] += d["bin_count"][bin2]
            d["bin_boundary"][bin1] = d["bin_boundary"][bin2]
            for i in range(bin2,max_idx-1):
                d["bin_count"][i] = d["bin_count"][i+1]
                d["bin_boundary"][i] = d["bin_boundary"][i+1]

            # the for-loop above made space. Now, split the bin.
            if max_idx == len(d["bin_boundary"]):
                # The right-most bin has no boundary, and must
                # be treated specially.
                # Let us assume the very first bin_width as a decent
                # estimate of bin_width here as well
                w =  d["bin_boundary"][1] - d["bin_boundary"][0]
                d["bin_boundary"][max_idx-1] = d["bin_boundary"][max_idx-2] + w
            else:
                # The usual case is to split the bin in the middle
                d["bin_boundary"][max_idx-1] = (d["bin_boundary"][max_idx]+d["bin_boundary"][max_idx-1]) / 2
                
            d["bin_count"][max_idx-1] = d["bin_count"][max_idx] // 2
            if d["bin_count"][max_idx] % 2 == 0:
                d["bin_count"][max_idx] = d["bin_count"][max_idx] // 2
            else:
                d["bin_count"][max_idx] = 1 + d["bin_count"][max_idx] // 2

        # the merged bin are to the right of max_idx
        else:
            d["bin_count"][bin2] += d["bin_count"][bin1]
            for i in range(bin1-1,max_idx,-1):
                d["bin_count"][i+1] = d["bin_count"][i]
                d["bin_boundary"][i+1] = d["bin_boundary"][i]
            try:
                d["bin_boundary"][max_idx+1] = d["bin_boundary"][max_idx]
            except:
                print("XX")
                print(max_idx)
                print(min_idx)
                print(bin1)
                print(bin2)
                raise
            d["bin_boundary"][max_idx] = (d["bin_boundary"][max_idx]+d["bin_boundary"][max_idx-1]) / 2
            d["bin_count"][max_idx+1] = d["bin_count"][max_idx] // 2
            if d["bin_count"][max_idx] % 2 == 0:
                d["bin_count"][max_idx] = d["bin_count"][max_idx] // 2
            else:
                d["bin_count"][max_idx] = 1 + d["bin_count"][max_idx] // 2
        return (max_idx,bin1,bin2)
    else:
        return None



def find_bins( d ):
    d["lcl_bin"] = None
    d["med_bin"] = None
    d["ucl_bin"] = None

    acc = 0
    for b in range(d["num_bins"]):
        acc += d["bin_count"][b]
        if (d["lcl_bin"] is None) and (d["lcl_idx"] < acc):
            d["lcl_bin"] = b
        if (d["med_bin"] is None) and (d["med_idx_1"] < acc):
            assert d["med_idx_2"] < acc
            d["med_bin"] = b
        if (d["ucl_bin"] is None) and (d["ucl_idx"] < acc):
            d["ucl_bin"] = b

    return (d["lcl_bin"], d["med_bin"], d["ucl_bin"])



def recount_bins( d ):
    (_,l) = d["obs"].shape
    for i in range(len(d["bin_count"])): d["bin_count"][i]=0
    n=0
    for i in range(1,l):
        for j in range(i+1,l):
            n += 1
            slope = float(d["obs"][1][j]-d["obs"][1][i]) / float(d["obs"][0][j]-d["obs"][0][i])
            record_value( d, slope, 1 )
            if d["trace"] and (n%10000000) == 0:
                print( "recount_bins: done "+str(n//1000000)+" m")

    d["n"] = n
    (lcl_bin, med_bin, ucl_bin) = find_bins( d )

    if d["trace"]:
        print("recount_bins:")
        print("Bin counts: " + str(d["bin_count"]) )
        print("Bin boundaries: " + str(d["bin_boundary"]) )
        print("Median (idx: "+str(d["med_idx_1"])+","+str(d["med_idx_1"])+") is in " + str(d["med_bin"]) + ", lcl (idx: "+str(d["lcl_idx"])+") in " + str(d["lcl_bin"]) + ", ucl (idx: "+str(d["ucl_idx"])+") in " + str(d["ucl_bin"]))

    return (lcl_bin, med_bin, ucl_bin)



def populate_bins( d ):
    (_,l) = d["obs"].shape
    low = d["lcl_bin"]
    med = d["med_bin"]
    high= d["ucl_bin"]

    if d["bin_count_accurate"]:
         d["lo"] = numpy.full( d["bin_count"][low], numpy.inf )
         d["me"] = numpy.full( d["bin_count"][med], numpy.inf )
         d["hi"] = numpy.full( d["bin_count"][high],numpy.inf )
    else:
         d["lo"] = numpy.full( d["max_size"], numpy.inf )
         d["me"] = numpy.full( d["max_size"], numpy.inf )
         d["hi"] = numpy.full( d["max_size"],numpy.inf )

    lo_ptr = 0
    me_ptr = 0
    hi_ptr = 0

    for i in range(len(d["bin_count"])): d["bin_count"][i]=0
    
    # This function tries to populate the bins while also maintaining bin counts.
    # Since the first round is based on estimates, it might happen that
    # (a) too much data falls into a bin and bin boundaries need to be re-done; or
    # (b) that the wrong bins have been selected for population.
    # If anything goes wrong, we stop populating but finish the counting, so that
    # we can get it right the next time around.

    still_good = True
    
    for i in range(0,l):
        for j in range(i+1,l):
            slope = float(d["obs"][1][j]-d["obs"][1][i]) / float(d["obs"][0][j]-d["obs"][0][i])
            bin = record_value( d, slope, 1 )

            if still_good and (bin == low):
                try:
                    d["lo"][lo_ptr] = slope
                except:
                    # This bucket is full. Stop populating.
                    still_good = False
                    if d["trace"]:
                        print( "Polulated bins with: low ("+str(low)+"): "+str(lo_ptr)+\
                               " med ("+str(med)+"): "+str(me_ptr)+\
                               " high ("+str(high)+")"+str(hi_ptr) )
                lo_ptr += 1
                if d["trace"] and False:
                    print( "Added "+str(slope)+" to low bin ("+str(bin)+"). New len: "+str(lo_ptr) )
                    if (d["bin_boundary"][bin-1] >= slope) or (d["bin_boundary"][bin] <= slope):
                        print("Wrong: "+str(d["bin_boundary"][bin-1])+","+str(d["bin_boundary"][bin]))

            if still_good and (bin == med):
                try:
                    d["me"][me_ptr] = slope
                except:
                    # This bucket is full. Stop populating.
                    still_good = False
                    print( "Polulated bins with: low ("+str(low)+"): "+str(lo_ptr)+\
                           " med ("+str(med)+"): "+str(me_ptr)+\
                           " high ("+str(high)+")"+str(hi_ptr) )
                me_ptr += 1
                if d["trace"] and False:
                    print( "Added "+str(slope)+" to med bin ("+str(bin)+"). New len: "+str(me_ptr) )
                    if (d["bin_boundary"][bin-1] >= slope) or (d["bin_boundary"][bin] <= slope):
                        print("Wrong: "+str(d["bin_boundary"][bin-1])+","+str(d["bin_boundary"][bin]))

            if still_good and (bin == high):
                try:
                    d["hi"][hi_ptr] = slope
                except:
                    # This bucket is full. Stop populating.
                    still_good = False
                    print( "Polulated bins with: low ("+str(low)+"): "+str(lo_ptr)+\
                           " med ("+str(med)+"): "+str(me_ptr)+\
                           " high ("+str(high)+")"+str(hi_ptr) )
                hi_ptr += 1
                if d["trace"] and False:
                    print( "Added "+str(slope)+" to high bin ("+str(bin)+"). New len: "+str(hi_ptr) )
                    if (d["bin_boundary"][bin-1] >= slope) or (d["bin_boundary"][bin] <= slope):
                        print("Wrong: "+str(d["bin_boundary"][bin-1])+","+str(d["bin_boundary"][bin]))

    if still_good:

        d["low_len"] = lo_ptr
        d["med_len"] = me_ptr
        d["high_len"]= hi_ptr
        d["low_bin"] = low
        d["med_bin"] = med
        d["high_bin"]= high

        # The arrays might not be full, but have been initialized
        # with numpy.inf, so all unused elements will sort to the right
        # of all used elements.
        if d["trace"]: print( "Sorting" )
        d["lo"].sort()
        d["me"].sort()
        d["hi"].sort()

        if d["trace"]:
            print( "Populated bins with: low ("+str(low)+"): "+str(lo_ptr)+\
                   " med ("+str(med)+"): "+str(me_ptr)+\
                   " high ("+str(high)+")"+str(hi_ptr) )

    return still_good



def get_percentiles( d ):
    n1 = d["bin_count"].sum()
    n = d["n"]
    if n1 != n:
        print( 'd["bin_count"].sum(): '+str(d["bin_count"].sum()) )
        print( "n: "+str(n) )

    idx_f  = d["lcl_idx"]
    idx_low_1 = int(idx_f)
    if idx_f.is_integer(): idx_low_2 = idx_low_1
    else:                  idx_low_2 = idx_low_1 + 1
        
    idx_f  = 0.5 * float(n)
    idx_med_1 = int(idx_f)
    if idx_f.is_integer(): idx_med_2 = idx_med_1
    else:                  idx_med_2 = idx_med_1 + 1

    idx_f  = d["ucl_idx"]
    idx_high_1 = int(idx_f)
    if idx_f.is_integer(): idx_high_2 = idx_high_1
    else:                  idx_high_2 = idx_high_1 + 1

    if n < 2:
        assert False

    for b in range(0,d["low_bin"]):
        idx_low_1  -= int( d["bin_count"][b] )
        idx_low_2  -= int( d["bin_count"][b] )
        idx_med_1  -= int( d["bin_count"][b] )
        idx_med_2  -= int( d["bin_count"][b] )
        idx_high_1 -= int( d["bin_count"][b] )
        idx_high_2 -= int( d["bin_count"][b] )
    value_low = (d["lo"][idx_low_1]+d["lo"][idx_low_2]) / 2

    for b in range(d["low_bin"],d["med_bin"]):
        idx_med_1  -= int( d["bin_count"][b] )
        idx_med_2  -= int( d["bin_count"][b] )
        idx_high_1 -= int( d["bin_count"][b] )
        idx_high_2 -= int( d["bin_count"][b] )
    value_med = (d["me"][idx_med_1]+d["me"][idx_med_2]) / 2
    
    for b in range(d["med_bin"],d["high_bin"]):
        idx_high_1 -= int( d["bin_count"][b] )
        idx_high_2 -= int( d["bin_count"][b] )
    value_high = (d["hi"][idx_high_1]+d["hi"][idx_high_2]) / 2
        
    return (value_low,value_med,value_high)

