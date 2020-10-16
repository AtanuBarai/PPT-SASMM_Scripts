"""
A fast algorithm to calculate reuse distance 
Uses partial sums and a self-balancing AVL tree 
from the intervaltree module

"""

import sys
from intervaltree import Interval, IntervalTree
trace_file = sys.argv[1]
import numpy as np
num_lines = sum(1 for line in open(trace_file))
sd_vals = np.zeros(num_lines,dtype='int')

outfile = open('sd_vals.out','w')


access_time = {}
T = IntervalTree()

import cPickle as cp 
def checkpoint(data):
    cp.dump(data,open("sd_vals.pickle",'w'))

def get_distance(last_access_time,now):
    #print "Inserting interval ",last_access_time,last_access_time+1
    T.add(Interval(last_access_time,last_access_time+1))
    #T.print_structure()
    num_holes = get_holes(last_access_time, T.top_node)
    return now - last_access_time - num_holes


#def get_interval(node):
#    return next(iter(node.s_center))

def get_holes(t, node):
    """
    function to return the number of holes in front
    of (i.e after) time t
    """
    iv = node.get_interval()
    if t < iv.begin:
        assert(not node.left_node is None)
        # go left. add partial sum of right branch
        return iv.length() + node.right_sum + get_holes(t,node.left_node)
    else:
        if t >= iv.end:
            assert(not node.right_node is None)
            # go right.
            return get_holes(t, node.right_node)
            
        else:
            assert ( t >= iv.begin and t < iv.end)
            return iv.end - t + node.right_sum
            

with open(trace_file) as tf:
    t = 0
    for address in tf:
        address = address[:-1]
        try:
            last_access_time = access_time[address]
            sd = get_distance(last_access_time,t)
            assert (sd >= 0)
            access_time[address] = t
        except KeyError:
            access_time[address] = t
            sd = -1
        sd_vals[t] = sd
        #print(sd_vals[t], file=outfile)
        print >>outfile, sd
        t += 1
        if ( t% 10000 == 0):
            print "finished %d traces" %(t)


checkpoint(sd_vals)
#print sd_vals
#cp.dump(sd_vals,open("sd_vals.pickle",'w'))
