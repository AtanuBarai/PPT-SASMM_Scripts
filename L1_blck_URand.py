"""
Author          :       Yehia Arafa
Last Modified   :       
File            :       L1_blck_URand.py
Purpose         :       
Description     :       
Output          :       
"""

import sys, re, os, random
import numpy as np
import math

'''
#VOLTA V100
cache_size = 128*1024
line_size = 32
associativity = 170
num_sm = 80
max_num_blcks = 32

#VOLTA TITAN-V
cache_size = 96*1024
line_size = 32
associativity = 128
num_sm = 80
max_num_blcks = 32

#TURING TITAN-RTX
cache_size = 64*1024
line_size = 32
associativity = 86
num_sm = 72
max_num_blcks = 32

#Ampere RTX 3070
cache_size = 96*1024
line_size = 32
associativity = 128
num_sm = 48
max_num_blcks = 32

#Design Space Exploration
cache_size = 48*1024
line_size = 32
associativity = 6
num_sm = 16
max_num_blcks = 32
'''


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
        self.sp = {} #dictionary of stack pointers

    def push(self,address,t,el=None):
        if not el is None:
            se = el #print "pushing ",el
        else:
            se = stack_el((address,t)) #print "pushing ",address,t

        if not self.sp == {}:  #stack is not empty
            se.next_el = self.sp["top"]
            se.prev_el = None
            self.sp["top"].prev_el = se

        else: #stack empty
            se.next_el = None
            se.prev_el = None

        self.sp["top"] = se
        self.sp[se.access_time] = se

    def update(self,last_access,now,address):
        try:
            se = self.sp.pop(last_access) #pop deletes key, and returns value
            assert(se.address == address)
            assert(se.access_time == last_access)
            d = 0 #calculate distance from se to top of stack
            tmp = se
            while (not tmp.prev_el is None):
                d += 1
                tmp = tmp.prev_el
            if not se.prev_el is None: #remove from linked list
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


def usuage():
    print "[USUAGE]: "+__file__+" Memory trace"


def interleave_trace(smi_trace):
    '''
    Return the a warp level interleaved trace
    '''
    final_trace=[]
    interleaved_trace = [val for tup in zip(*smi_trace) for val in tup]
    for traces in interleaved_trace:
        final_trace = final_trace + traces.split(",")
    return final_trace


'''
def get_sm_reuse_prof_fast(smi_trace):
    #Return the stack distances (sd) the memory trace (time complexity: n^2)
    sd = []
    for addr, idx in zip(smi_trace, range(len(smi_trace))):
        window_trace = smi_trace[:idx]
        dict_sd = {}
        addr_found = False
        for w_adx in range(0,len(window_trace)):
            w_addr = window_trace[-w_adx -1]
            if addr == w_addr:
                addr_found = True
                break            
            dict_sd[w_addr] = True
        if addr_found: sd.append(len(dict_sd))
        else: sd.append(-1)
    return sd
'''


def get_sm_reuse_prof_fast(interleaved_trace):
    '''
    Return the stack distances (sd) the memory trace (time complexity: nlogn)
    '''
    access_time = {} 
    sd_vals = []#np.empty(len(interleaved_trace),dtype='int')
    t = 0
    sd_stack = Stack()
    for line in interleaved_trace:
        if line:
            line = line.split(" : ")
            opcode = int(line[0])
            address = line[1]
            #if opcode == 0:
            try:
                last_access_time = access_time[address]
                sd = sd_stack.update(last_access_time,t,address)
                access_time[address] = t
            except KeyError:
                access_time[address] = t
                sd_stack.push(address,t)
                sd = -1
            sd_vals.append(sd)
            t += 1
    return sd_vals


def get_prob_count_sd_sm_i(sd_vals):
    '''
    Return the Reuse Profile: unique stack distances (SD) and their probabilities (p(sd/SMi)))
    and counts for a given SM
    '''
    len_sd = len(sd_vals)
    uniq_sd,counts = np.unique(sd_vals,return_counts=True)
    p_uniq_sd_smi = map(lambda x: x/float(len_sd),counts) #Compute probabilities
    #print 'Sum of p(sd/smi) : ', np.sum(p_uniq_sd_smi)
    return zip(uniq_sd, p_uniq_sd_smi, counts)
 

def get_hit_rate_sm_i_naive(final_reuse_profile):
    '''
    Return the hit rates given RD and cache size only using the count in the RP: assumes fully assosiative cache 
    '''
    hit=0
    miss=0
    for items in final_reuse_profile:
        stack_distance = items[0]
        count = items[2]
        if stack_distance<0 or stack_distance >= cache_size:
            miss += count
        else:
            hit += count
    return (((hit * 1.0) / (miss + hit)) * 100)  


def get_hit_rate_sm_i_analytical(final_reuse_profile):
    '''
    Return the hit rate for each SM given RP, cache size, line size, and associativity
    '''
    np.seterr(all = 'raise')
    phit = 0.0 #Sum (probability of stack distance * probability of hit given D)
    num_blocks = (1.0 * cache_size)/line_size  # B = Num of Blocks (cache_size/line_size)
    
    for items in final_reuse_profile:
        stack_distance = items[0]
        probabilities = items[1]
        phit_given_d = 0.0 #To compute probability of a hit given D
        if stack_distance == -1:   phit_given_d = 0
        elif stack_distance == 0:  phit_given_d = 1.0
        else:
            for a in xrange(int(associativity)):
                try:
                    term_1 = ncr(stack_distance, a)
                    term_2 = math.pow((associativity/num_blocks), a)
                    term_3 = math.pow((1 - (associativity/num_blocks)), (stack_distance - a))
                    phit_given_d += (term_1 * term_2 * term_3)
                except FloatingPointError:
                    continue
	try:        
		phit += probabilities * phit_given_d
	except FloatingPointError:
                    phit =0 
    return (phit*100)
  

def ncr(n, m):
    '''
    n choose m
    '''
    if(m>n): return 0
    r = 1
    for j in xrange(1,m+1):
        try:
            r *= (n-m+j)/float(j)
        except FloatingPointError:
            continue
    return r



def main(trace_file):
    
    outF = open(str(os.path.splitext(trace_file)[0])+"_L1_results_URand.txt", "w")
    counter = 0
    queue=[[] for i in range(num_sm)]
    hit_rates=[]
    orig_trace = open(trace_file,'r').read().strip().split('\n\n') # list of trace table
    print >>outF,"*** Reuse Profiles & hit rate for different SMs: L1 Caches ***"
    fewer_blocks=0 #0->random, 1->RR 

    if len(orig_trace) < num_sm:
        fewer_blocks = 1
    for smi_trace in orig_trace:
        temp =[]
        smi_trace = smi_trace.split("----------------------------------- ")[1].split("\n=====\n")
        blck_id = smi_trace[0]
        smi_trace.pop(0)
        #if fewer_blocks == 1:
        #    index = int(blck_id)%num_sm
        #else:
        index = random.randint(0, num_sm-1)
        if len(queue[index]) < max_num_blcks:
            for i in range (len(smi_trace)):
                temp = temp + smi_trace[i].split("\n")
            queue[index].append(temp)

    for smi_trace in queue:
        if smi_trace:
            print >>outF,"SM "+str(counter)+":"
            counter+=1
            #if fewer_blocks:
                #interleaved_trace = smi_trace[0]
            #else:
            interleaved_trace = interleave_trace(smi_trace)
            rdi = get_sm_reuse_prof_fast(interleaved_trace)
            rpi = get_prob_count_sd_sm_i(rdi)
            print >>outF,"Reuse Profile:"
            for items in rpi:
                print >>outF,str(items[0]) +", "+ str(items[1])+", "+ str(items[2])
            print >>outF,"Hit Rate:" 
            #hit_rate_i = get_hit_rate_sm_i_naive(rpi)
            hit_rate_i = get_hit_rate_sm_i_analytical(rpi) 
            hit_rates.append(hit_rate_i)
            print >>outF,str(hit_rate_i) + str("%")
            print >>outF," "
            print >>outF,"-----------------------------"
            print >>outF," "

    print >>outF,"*********************************" 
    print >>outF,str("Min Hit Rate for ")+str(len(hit_rates))+str(" SMs: ")+str(min(hit_rates)) + str("%")
    print >>outF,str("Max Hit Rate for ")+str(len(hit_rates))+str(" SMs: ")+str(max(hit_rates)) + str("%")
    print >>outF,str("Average Hit Rate for ")+str(len(hit_rates))+str(" SMs: ")+str(sum(hit_rates)/len(hit_rates)) + str("%")
    print >>outF,"*********************************"
    print "Results Done"
    outF.close()

    
    
if __name__ == "__main__":
    if(len(sys.argv) != 2):   usuage()
    else:    
        main(sys.argv[1])
