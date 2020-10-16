"""
script to calculate re-use distances from a memory trace
"""

import sys
trace_file  = sys.argv[1]
access_time = {}

def readInChunks(fileObj, chunkSize=2048*1024):
    """
    Lazy function to read a file piece by piece.
    Default chunk size: 2MB.
    """
    while True:
        data = fileObj.read(chunkSize)
        if not data:
            break
        yield data


#def key1(node):
#    return node.x_center
#
#def key2(node):
#    return next(iter(node.s_center)).end

#def get_interval(node):
#    return next(iter(node.s_center))

#def dist(node, t):
#    if t < key1(node):
#        if not node.left_node is None:
#            return node.sum  + dist(node.left_node)
#        else:
 
#def dist(node, p):
#    iv = get_interval(node)
#    k1 = iv.begin
#    k2 = iv.end
#    if p < k1:
#        #continue search left
#        if not node.left_node is None:
#            return iv.data + iv.length() + dist(node.left_node,p)
#        else:
#            #cant't continue left*/
#            #shouldn't happen, since we have already added the interval.
#            assert(False)
#    else if p >= k2 : #need >= since the end-point is not included in the interval
#        # continue search right
#        if not node.right_node is None:
            
        
class stack_el(object):
    def __init__ (self,data): #next_el=None,prev_el=None):
        self.address = data[0]
        self.access_time = data[1]
        self.next_el = None
        self.prev_el = None

    def __str__(self):
        return "(%s %d)" %(self.address,self.access_time)


class Stack(object):
    def __init__ (self):
        self.elements = []
        #dictionary of stack pointers
        self.sp = {}

    def push(self,address,t,el=None):

        if not el is None:
            #print "pushing ",el
            se = el
        else:
            #print "pushing ",address,t
            se = stack_el((address,t))

        if not self.sp == {}:  #stack is not empty
            se.next_el = self.sp["top"]
            se.prev_el = None
            self.sp["top"].prev_el = se

        else: #stack empty
            se.next_el = None
            se.prev_el = None

        self.sp["top"] = se
        self.sp[se.access_time] = se
        #self.print_stack()


    def print_stack(self):
        import time
        print "printing stack"
        if self.sp == {}:
            print "[]"
            return
        else:
            se = self.sp["top"]
            while (True):#not se.next_el is None):
                print se ,
                #time.sleep(1)
                se = se.next_el
                if se is None:
                    break
            print 

    def update(self,last_access,now,address):
        try:
            se = self.sp.pop(last_access) #pop deletes key, and returns value
            assert(se.address == address)
            assert(se.access_time == last_access)

            #calculate distance from se to top of stack
            d = 0
            tmp = se
            while (not tmp.prev_el is None):
                d += 1
                tmp = tmp.prev_el

            #remove from linked list
            if not se.prev_el is None:
                se.prev_el.next_el = se.next_el
            else: 
                #the element is already at the top of the stack. 
                #just update access time and return depth
                se.access_time = now
                self.sp[se.access_time] = se
                return d

            if not se.next_el is None:
                se.next_el.prev_el = se.prev_el

            #update access time
            se.access_time = now
            #create an entry for popped element at the top of the stack
            self.push(None,None,el=se)
            #return the depth  of this element before it was brought to the top
            return d

        except KeyError:
            print "internal error. (%s %d) not found in dictionary" %(address,last_access)
            quit()
            

import cPickle as cp 
def checkpoint(data):
    cp.dump(data,open("sd_vals.pickle",'w'))

           
import numpy as np
num_lines = sum(1 for line in open(trace_file))
sd_vals = np.zeros(num_lines,dtype='int')

t = 0
sd_stack = Stack()
m = 0

with open(trace_file) as tf:
    for address in tf:
        address = address[:-1]
        # if not address.startswith('BB '):
        try:
            last_access_time = access_time[address]
            sd = sd_stack.update(last_access_time,t,address)

            #sd_vals[m] = sd
            #m += 1

            #print "stack distance is ",sd
            access_time[address] = t

        except KeyError:
            access_time[address] = t
            sd_stack.push(address,t)
            sd = -1
    
        sd_vals[t] = sd
        t += 1
        if ( t% 10000 == 0):
            print "finished %d traces" %(t)
        if ( t % 100000 == 0 ):
            print "checkpointing"
            checkpoint(sd_vals[0:t])
        #else: print "Not updated the stack : ", address

checkpoint(sd_vals)

#sd_vals = sd_vals[0:m]
#print sd_vals
#quit()

#print len(access_time.keys())


#print stats
print "Stack Distance Stats"
print "mean %f min %f max %f std %f" %(np.mean(sd_vals),np.min(sd_vals),np.max(sd_vals),np.std(sd_vals))
