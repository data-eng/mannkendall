import sys
import numpy
import time
import operator
import collections


def initializer( obs, max_size ):
    retv = {}
    retv["data"] = linkedlist = collections.deque()
    retv["max_size"] = max_size
    retv["left_heap"] = 0
    retv["right_heap"] = 0
    retv["trace"] = False

    if obs is not None:
        (_,l) = obs.shape
        min = None
        max = None
        for i in range(1,l):
            slope = float(obs[1][i]-obs[1][i-1]) / float(obs[0][i]-obs[0][i-1])
            try:
                (_,_,s) = min
                if slope < s: min = (i-1,i,slope)
            except TypeError:
                min = (i-1,i,slope)
            try:
                (_,_,s) = max
                if slope > s: max = (i-1,i,slope)
            except TypeError:
                max = (i-1,i,slope)
        retv["data"].append( min )
        retv["data"].append( max )

    return retv



def find_place( d, item ):
    (_,_,v) = item
    retv = -1
    for (i,j,x) in d["data"]:
        retv += 1
        if d["trace"]:
            print( "XX: v=" + str(v) + " cur idx=" + str(retv) + " cur value:" + str(x) ) 
        if v < x:
            if d["trace"]:
                print( "XX1: Found " + str(retv) + " " + str(x) ) 
            break
    return retv



def insert( d, item ):
    pos = find_place( d, item )
    if d["trace"]:
        print ( "POS: " + str(pos) )
    if len(d["data"]) < d["max_size"]:
        d["data"].insert( pos, item )
    else:
        (_,_,v) = item
        (_,_,vleft) = d["data"][0]
        (_,_,vright) = d["data"][-1]
        if v <= vleft:
            d["left_heap"] += 1
        elif v >= vright:
            d["right_heap"] += 1
        else:
            if d["right_heap"] < d["left_heap"]:
                d["data"].pop()
                d["right_heap"] += 1
            else:
                d["data"].popleft()
                d["left_heap"] += 1
            d["data"].insert( pos, item )



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

all_slopes = []


def generator( obs ):
    (_,l) = obs.shape
    #for i in range(1,l):
    for i in range(l-1,0,-1):
        for j in range(i+1,l):
        #for j in range(l-1,i,-1):
            slope = float(obs[1][j]-obs[1][i]) / float(obs[0][j]-obs[0][i])
            all_slopes.append( (i,j,slope) )


#data = [ [0,10,20,30,40,50,60,70,80,90,100,110,120], [5,1,5,6,5,2,4,8,2,3,1,2,4] ]
#datastack = numpy.array(data)

obs = numpy.loadtxt( sys.argv[1] ).T

print( "loaded " + str(obs.shape) )
obsT = obs.T
good = ((obsT)[~numpy.isnan(obsT).any(axis=1)]).T
print( "after nan rm " + str(good.shape) )


#a=datastack[:,datastack[1,:].argsort()]
#print(datastack)
generator( good )

print("Gen")

all_sorted = sorted(all_slopes, key = operator.itemgetter(2))
#print(all_sorted)

print("Sort")

d = initializer( None, 50*1024 )
#print( d )

print("Init")

i=0
t0 = time.time()
for x in all_slopes:
    insert( d , x )
    if( i%10000 == 0 ):
        dt = time.time() - t0
        print( "Insert: "+str(i)+"/"+str(len(all_slopes)) )
        print( str(d["left_heap"]) + " - " + str(d["right_heap"]) )
        (_,_,vleft) = d["data"][0]
        (_,_,vright) = d["data"][-1]
        print( str(vleft) + " - " + str(vright) )
        print( "med: " + str(median(d)) )
        print( "dt: " + str(dt) )
        t0 = time.time()
    i+=1
    #print( d )
#print( d )

print( d["left_heap"] )
print( d["right_heap"] )
print( median(d) )

#d.append( all_sorted[0] )
#d.append( all_sorted[-1] )


