"""
Author          :       Atanu Barai, Gopinath Chennupati
Last Modified   :       
File            :       reuse_profile_bb_i_shared_mem_Ver5.py
Purpose         :       Preprocess the raw trace resulted out of hacked Byfl
                        Compute the reuse distribution of a given BasicBlock
                        Compute p(sd/BBi)
Description     :       Sample (10 or 20 or 25) the window sizes in order to speedup the process
Output          :       Write the sd,p(sd/BBi) into a file
"""

import sys, re, io
import cPickle as cp 
import numpy as np
import random, time
import gc
import math
import copy
from operator import itemgetter
from collections import OrderedDict
import matplotlib.pyplot as plt

distributions = ['_uniform', '_round_robin']  # _round_robin / _uniform / _normal

# distributions = ['_round_robin']
sample_size = 5 #random number of sample


def usuage():
    print "[USUAGE]: "+__file__+" Basicblocks_table trace num_cores bb_probs"

def preprocess_orig_trace(bb_table,trace_file):
    '''
    Preprocess original trace to include BasicBlock names and their END
    '''
    orig_mem_trace = open(trace_file,'r').read().strip()
    #Fixing the irregular start of the basic blocks, meaning there are >=two consecutive BB starts before it ends
    start_idx = [match.start() for match in re.finditer('BB START: ', orig_mem_trace)]
    for i in range(0, len(start_idx)-1):
        s_id = start_idx[i]
        nl_id = orig_mem_trace[s_id:].find('\n')
        rand_id = s_id+nl_id+1
        #bb_rand_name = orig_mem_trace[s_id:rand_id]
        if start_idx[i+1] == rand_id: 
            #print(i, s_id, start_idx[i+1])
            rand_str = orig_mem_trace[s_id+len('BB START: '):rand_id]
            bb_done = 'BB DONE: '+rand_str
            orig_mem_trace = orig_mem_trace[:rand_id]+bb_done+orig_mem_trace[rand_id:]
            def add_soffset(_a):
                return _a+len(bb_done)
            start_idx[i:] = map(add_soffset, start_idx[i:])
    print('Length of original trace after BB START Prep {}'.format(len(orig_mem_trace)))
    #Fixing the irregular end of the basic blocks, meaning there are >=two consecutive BB ends
    end_idx = [match.start() for match in re.finditer('BB DONE: ', orig_mem_trace)]
    for i in range(0, len(end_idx)-1):
        curr_eid = end_idx[i]
        next_eid = end_idx[i+1]
        eid_line = curr_eid+orig_mem_trace[curr_eid:].find('\n')+1
        next_start_id = orig_mem_trace[curr_eid:next_eid].find('BB START: ')
        if next_start_id == -1: 
            rand_id = next_eid+orig_mem_trace[next_eid:].find('\n')+1
            rand_str = orig_mem_trace[next_eid+len('BB DONE: '):rand_id]
            bb_start = 'BB START: '+rand_str
            orig_mem_trace = orig_mem_trace[:eid_line]+bb_start+orig_mem_trace[eid_line:]
            def add_eoffset(_a):
                return _a+len(bb_start)
            end_idx[i:] = map(add_eoffset, end_idx[i:])
    print('Length of original trace after BB DONE Prep {}'.format(len(orig_mem_trace)))
    #print bb_table
    for b_t in bb_table:
        b_t_info = b_t.split(': ')
        #print b_t_info[0]+":", b_t_info[2]
        if b_t_info[1] in orig_mem_trace:            
            orig_mem_trace = orig_mem_trace.replace(b_t_info[1],b_t_info[2]) 
            
            orig_mem_trace = orig_mem_trace.replace('LOAD: ','')
            orig_mem_trace = orig_mem_trace.replace('STORE: ','')
            
    orig_mem_trace = orig_mem_trace.split('\n')    
    o_t_file = open('processed_trace.dat','w')
    for addr in orig_mem_trace:
        print >>o_t_file,addr
    print "************** Finished preprocessing the trace *************"
    return orig_mem_trace
    #return orig_mem_trace

def read_prep_trace_from_file(prep_trace_file):
    return open(prep_trace_file,'r').read().strip().split('\n')

def get_bb_i_windows(bbi_name, orig_trace):
    '''
    Return the window size (k,z) of a given BasicBlock    
    i = start of BBi
    j = end of BBi
    '''        
    
    bbi_wins = []
    bb_start_key = 'BB START: '+bbi_name
    bb_end_key = 'BB DONE: '+bbi_name
    #print bb_start_key, bb_end_key
    if bb_start_key in orig_trace:
        for i, x in enumerate(orig_trace):
            if x == bb_start_key:
                j = orig_trace[i:].index(bb_end_key)+i
                # if j != i+1:
                #     bbi_wins.append([i,j])
                bbi_wins.append([i,j])
    else:
        bbi_wins = [[0,0]]
        
    return bbi_wins


def get_bb_i_windows_with_mem_references(bbi_name, orig_trace):
    '''
    Return the window size (k,z) of a given BasicBlock    
    i = start of BBi
    j = end of BBi
    '''

    bbi_wins = []
    bb_start_key = 'BB START: '+bbi_name
    bb_end_key = 'BB DONE: '+bbi_name
    #print bb_start_key, bb_end_key
    if bb_start_key in orig_trace:
        print bb_start_key, 'in trace'
        for i, x in enumerate(orig_trace):
            if x == bb_start_key:
                j = orig_trace[i:].index(bb_end_key)+i
                if j != i+1:
                    bbi_wins.append([i, j])
                    # if list(filter(lambda addr: addr[:2] == '0x', orig_trace[i:j])):
                    #     bbi_wins.append([i,j])

    return bbi_wins

def sample_bbi_wins(bbi_win_sizes, sample_size):
    '''
    Randomly sample 'sample_size' number of bbi_win_sizes
    '''
    sampled_wins = []
    len_bbi_win_sizes = len(bbi_win_sizes)
    if len_bbi_win_sizes > sample_size: 
        indices = list(range(len_bbi_win_sizes))
        sample_indices = random.sample(indices,sample_size)
        '''
        if not len_bbi_win_sizes-1 in sample_indices: 
            sample_indices[-1] = len_bbi_win_sizes-1
        '''
        for s_idx in sample_indices:
            sampled_wins.append(bbi_win_sizes[s_idx])
    else: sampled_wins = bbi_win_sizes
    
    return sampled_wins

def get_bbi_reuse_prof_fast(bb_i_size, orig_trace):
    sd = [0,0]
    if bb_i_size[1] > bb_i_size[0]+1:
        bbi_trace = orig_trace[bb_i_size[0]+1:bb_i_size[1]]        
        sd = []     
        # Calculate the SDs for each addr in BBi
        for addr, idx in zip(bbi_trace,range(bb_i_size[0]+1,bb_i_size[1])):
            window_trace = orig_trace[:idx]
            dict_sd = {}
            addr_found = False
            for w_adx in range(0,len(window_trace)):
                w_addr = window_trace[-w_adx -1]
                if addr == w_addr:
                    addr_found = True
                    break
                if w_addr[:2]=='0x':
                    dict_sd[w_addr] = True
            if addr_found: sd.append(len(dict_sd))
            else: sd.append(-1)   
    sd = np.array(sd)
    return sd

def get_bb_i_trace(bb_i_start,bb_i_end,orig_trace):
    '''
    Return the memory trace of a given BasicBlock
    '''
    return orig_trace[bb_i_start:bb_i_end]

def get_prob_sd_bb_i(sd_vals):
    '''
    Return the unique reuse distances (D) and their probabilities (p(D/BBi)))
    for a given BasicBlock
    '''
    len_sd = len(sd_vals)
    uniq_sd,counts = np.unique(sd_vals,return_counts=True)
    #Compute probabilities
    p_uniq_sd_bbi = map(lambda x: x/float(len_sd),counts)
    print 'Sum of p(sd/bbi) : ', np.sum(p_uniq_sd_bbi)
    return zip(uniq_sd, p_uniq_sd_bbi, counts)    

def get_final_prob_bb_i(reuse_profile_bbi,bbi_prob):
    '''
    Multiply conditional probabilities of a given Basicblock with 'weighted probability'
    '''
    for i in range(0,len(reuse_profile_bbi)):
        reuse_profile_bbi[i][1] = float(bbi_prob) * float(reuse_profile_bbi[i][1])

    return reuse_profile_bbi


def get_all_bbi_reuse_profile(final_reuse_profile):
    '''
    Merge the reuse profile probabilities of duplicate Reuse Distances
    '''    
    res = {}    
    for item in final_reuse_profile:
        
    	if item[0] not in res:
            #res[item[0]] = [0.0]
    	    res[item[0]] = [0.0, 0.0]
    	res[item[0]][0] += item[1]
        res[item[0]][1] += item[2]
    res = OrderedDict(sorted(res.items(), key=itemgetter(0)))
    print "********** Done adding the duplicate probabilities ***********"
    return res

def isListEmpty(inList):
    if isinstance(inList, list): # Is a list
        return all( map(isListEmpty, inList) )
    return False # Not a list

def replace_with_interleaved_trace(bbi_name, orig_trace, interleaved_bb_traces):
    '''
    Takes basic block name, original trace, remove basic blocks' 
    memory trace from the original trace and insert interleaved
    memory trace's memory address in that place
    '''
    bb_start_key = 'BB START: '+bbi_name
    bb_end_key = 'BB DONE: '+bbi_name
    #print bb_start_key, bb_end_key

    if bb_start_key in orig_trace:        
        for i, x in enumerate(orig_trace):
            if x == bb_start_key:                
                j = orig_trace[i:].index(bb_end_key)+i                
                if j is not i+1:
                    i = i + 1
                    del orig_trace[i:j]
                    if interleaved_bb_traces:
                        # del orig_trace[i:j]
                        for addr in interleaved_bb_traces.pop(0):
                            orig_trace.insert(i, addr)
                            i = i + 1
    return orig_trace

def is_blank_bb(bbi_windows, orig_trace):
    '''
    Check if any basicblock doesn't have any memory trace between bb_start and bb_end
    '''
    if(bbi_windows == [[0,0]] or len(bbi_windows) == 0):
        return True
    
    # bb_references = map(lambda x: int(x, 16), list(filter(
    #     lambda addr: addr[:2] == '0x', orig_trace[bbi_windows[0][0]: bbi_windows[0][1]])))
    bb_references = []
    for bb_window in bbi_windows:
        bb_references.extend = map(lambda x: int(x, 16), list(filter(
            lambda addr: addr[:2] == '0x', orig_trace[bb_window[0]: bb_window[1]])))
        if len(bb_references) != 0:
            return False    
    return True

def normalize_probs(num_cores):
    return num_cores


def plot_graph(x, y, num_cores, distri):
    num_ticks = 20
    x_pos = np.arange(len(x))
    tick_interv = len(x) / num_ticks

    plt.bar(x_pos, y, color='green')

    plt.xticks(x_pos[::tick_interv], x[::tick_interv], rotation='vertical')
    # plt.locator_params(axis='x', nbins=10)
    plt.autoscale(enable=True, axis='y')    
    plt.xlabel("D")
    plt.ylabel("log(P(D))")
    plt.title("Reuse_profile_" + str(num_cores) + "_cores" + distri)
    plt.gca().set_yscale('log')
    plt.tight_layout()
    plt.savefig("fig_reuse_profile_mimic_" + str(num_cores) + "_cores" + distri + ".png")


def compute_reuse_profile(bb_table, target_mem_trace, bb_probs, num_cores, distri):
    print "\n\n***************** Computing Reuse Profile of the given trace *****************"
    
    final_reuse_profile = []
    # final_reuse_prof_crit_sec = []
    final_reuse_prof_parallel_sec = []
    sum_pbbs = 0
    for i in range(0,len(bb_table)):
        bb_table_row = bb_table[i]        
        b_t_info = bb_table_row.split(': ')
        print "\n------------> Processing BasicBlock ", b_t_info[2], ", Probability : ", bb_probs[i]
        if bb_probs[i] == 0.0:
            print "BB with probability 0.0., skipping"
            continue
        sum_pbbs += bb_probs[i]
        bbi_windows = get_bb_i_windows_with_mem_references(
            b_t_info[2], target_mem_trace)
        if len(bbi_windows) == 0:
            print "Blank BB, skipping"
            continue
        bbi_win_sizes = sample_bbi_wins(bbi_windows, sample_size)
        res_sds = np.array([])
        for bb_i_size in bbi_win_sizes:
            res_sds = np.concatenate([res_sds, get_bbi_reuse_prof_fast(bb_i_size,target_mem_trace)])
        zipped_sd_psd_i = get_prob_sd_bb_i(res_sds)
        res_sd_psd_i = [list(sd_psd) for sd_psd in zipped_sd_psd_i]
        f_res_sd_psd_i = get_final_prob_bb_i(res_sd_psd_i, bb_probs[i])
        for f_sd_psd_i in f_res_sd_psd_i:
            final_reuse_profile.append(f_sd_psd_i)
            if("OUT__" in b_t_info[2]):
                final_reuse_prof_parallel_sec.append(f_sd_psd_i)

        print "Stack Distance Stats of BasicBlock ", b_t_info[2]
        print "mean %f min %f max %f std %f" %(np.mean(res_sds),np.min(res_sds),np.max(res_sds),np.std(res_sds))
    #Get the final reuse profile for the program by merging the probabilities for all the BBi
    #print final_reuse_profile
    all_bbi_sd_profile = get_all_bbi_reuse_profile(final_reuse_profile)
    # crit_bbi_sd_profile = merge_reuse_profile(final_reuse_prof_crit_sec)
    parallel_bbi_sd_profile = get_all_bbi_reuse_profile(
        final_reuse_prof_parallel_sec)
    print "\nSize of the final reuse profile : ", len(all_bbi_sd_profile)
    
    sum_f = sum(value[0] for key,value in all_bbi_sd_profile.iteritems())    
    stack_dist = []
    probability_sd = []
    with open("reuse_profile_mimic_"+str(num_cores) + "_cores" + distri + ".dat", "w") as f:
        for key,value in all_bbi_sd_profile.iteritems():
            stack_dist.append(key)
            probability_sd.append(value[0])
            f.write(str(key).ljust(4)+" , "+str(value[0]).ljust(20)+" , "+str(value[1]).ljust(4)+"\n")
            #f.write(str(key).ljust(4)+" , "+str(value[0]).ljust(20)+"\n")
    # plot_graph(stack_dist, probability_sd, num_cores, distri)
    # with open("reuse_profile_crit_sec_"+ distri +".dat","w") as f:
    #     for key,value in crit_bbi_sd_profile.iteritems():
    #         f.write(str(key).ljust(4)+" , "+str(value[0]).ljust(20)+" , "+str(value[1]).ljust(4)+"\n")
    #         #f.write(str(key).ljust(4)+" , "+str(value[0]).ljust(20)+"\n")
    

    with open("reuse_profile_parallel_sec_"+str(num_cores)+ "_cores" + distri + ".dat", "w") as f:
        for key,value in parallel_bbi_sd_profile.iteritems():
            f.write(str(key).ljust(4)+" , "+str(value[0]).ljust(20)+" , "+str(value[1]).ljust(4)+"\n")
            #f.write(str(key).ljust(4)+" , "+str(value[0]).ljust(20)+"\n")

    print "Finished preparing the final reuse_profile of the program"
    print "Sum of p(BBi) : ", sum_pbbs
    print "Final sum of the reuse profile probabilities", sum_f 
    return all_bbi_sd_profile

def merge_traces(*argv):  
    # trace and core being used interchangeably
    bb_table = argv[0]
    first_trace = argv[1]    
    all_traces = argv[2]
    num_traces = argv[3]
    distri = argv[4]
    all_traces[0] = first_trace    
    all_trace_windows = [[]for _ in range(num_traces)]    
    merged_trace = []
    print "Merging", num_traces, " traces"
    for i in range(0,len(bb_table)):
        bb_table_row = bb_table[i]
        b_t_info = bb_table_row.split(': ')        
        if("OUT__" in b_t_info[2]):
            print bb_table_row
            interleaved_bb_traces = []
            num_bb_per_core = []
            for t_id in range (0, num_traces):
                all_trace_windows[t_id] = get_bb_i_windows(
                    b_t_info[2], all_traces[t_id])
                num_bb_per_core.append(len(all_trace_windows[t_id]))
            # num_bb_per_core = len(num_bb_per_core[0])
            # print all_trace_windows        
            # print num_bb_per_core
            
            for bb_idx_per_core in range (0, num_bb_per_core[0]):
                # print 'BB occurance ' + str(bb_idx_per_core) + '\n'

                mem_trace_per_core = [[]for _ in range(num_traces)]
                # print mem_trace_per_core, "\n------------------------------------"
                for trace in range (0, num_traces):                    
                    if(bb_idx_per_core < num_bb_per_core[trace]):
                        # print all_trace_windows[trace][bb_idx_per_core][0], all_trace_windows[trace][bb_idx_per_core][1]
                        for ind in range(all_trace_windows[trace][bb_idx_per_core][0]+1, \
                            all_trace_windows[trace][bb_idx_per_core][1]):
                                mem_trace_per_core[trace].append(
                                    all_traces[trace][ind])
                # print mem_trace_per_core, "\n------------------------------------"
                trace = 0
                interleaved_bb_traces.append([])
                while not isListEmpty(mem_trace_per_core):                    
                    #Generate Random Number
                    if distri == '_uniform':
                        # for uniform random
                        trace = random.randint(0, num_traces-1)
                    # elif distri == '_normal':
                    #     while True:
                    #         trace = np.random.normal(
                    #             (num_traces)/2, num_traces/2)
                    #         trace = math.floor(trace)
                    #         if trace >= 0 and trace < num_traces:
                    #             trace = int(trace)
                    #             break
                    
                    # Get the interleaved basic_blocks for each trace
                    if mem_trace_per_core[trace]:
                        interleaved_bb_traces[bb_idx_per_core].append(
                            mem_trace_per_core[trace].pop(0))
                    
                    # for round robin                    
                    if(distri == '_round_robin'):
                        trace += 1
                        if trace == num_traces:
                            trace = 0
            # print interleaved_bb_traces
            merged_trace = replace_with_interleaved_trace(
                b_t_info[2], all_traces[0], interleaved_bb_traces)
            
    # o_t_file = open('Interleaved_L3_trace_' + str(num_traces) + '_cores_' + distri +'.dat','w')
    # for addr in merged_trace:
    #    print >>o_t_file,addr
    return merged_trace

def generate_each_thread_trace_and_get_shared(bb_table, orig_trace, num_cores, differ, shared_variable_trace, \
    orig_bb_probs):
    print "Generating trace for each core"
    # shared_variable_trace = ['0x000000000000']
    all_core_trace = [[]for _ in range(num_cores)]
    all_core_windows = [[]for _ in range(num_cores)]
    sum_p_bbi_ser = 0.0
    sum_p_bbi_par = 0.0
    master_thread_prob = []
    other_thread_prob = []
    # all_core_pBB = [[]for _ in range(num_cores)]
    for i in range(0, len(bb_table)):
        bb_table_row = bb_table[i]
        print "\n\n-----------------------------------------------------------"
        print bb_table_row
        b_t_info = bb_table_row.split(': ')
        print "Calculating BBi Windows"
        bbi_windows = get_bb_i_windows(b_t_info[2],orig_trace)
        if bbi_windows == [[0, 0]]:
            print "This BB does not appear on the trace"
            continue
        # if "_ZL14OUT__2__6363__Pv, for.body3" in b_t_info[2]:
        #     print bbi_windows
        len_bbi_win_sizes = len(bbi_windows)
        print "LENGTH of basic block windows" ,len_bbi_win_sizes
        if len_bbi_win_sizes == 1:
            print bbi_windows
        # print " BBi Windows: ", bbi_windows
        min_size_bbi_windows_per_core = 1
        max_size_bbi_windows_per_core = 1
        remaining_windows = 0

        if("OUT__" in b_t_info[2]):
            sum_p_bbi_par = sum_p_bbi_par + orig_bb_probs[i]
            if(len_bbi_win_sizes < num_cores):
                min_size_bbi_windows_per_core = 1
            else:
                # The beginning threads will execute more loop iterations if number of iterations can not be evenly \
                # distributed
                min_size_bbi_windows_per_core = int(math.floor(
                    float(len_bbi_win_sizes) / float(num_cores)))
                max_size_bbi_windows_per_core = int(
                    math.ceil(float(len_bbi_win_sizes) / float(num_cores)))
                print "MAX ", max_size_bbi_windows_per_core, " MIN ", min_size_bbi_windows_per_core
                remaining_windows = len_bbi_win_sizes % num_cores

            if(len_bbi_win_sizes == 1):
                for core in range (0, num_cores):
                    all_core_windows[core].append(bbi_windows)
            else:
                max_core = len_bbi_win_sizes if len_bbi_win_sizes < num_cores else num_cores
                for core in range (0, max_core):
                    if(core < remaining_windows):
                        print "Core ", core, " In MAX"
                        #default openmp scheduler
                        all_core_windows[core].append(bbi_windows[:max_size_bbi_windows_per_core]) 
                        del bbi_windows[:max_size_bbi_windows_per_core]                        
                    else:
                        print "Core ", core, " In MIN"
                        #default openmp scheduler
                        all_core_windows[core].append(bbi_windows[:min_size_bbi_windows_per_core])
                        del bbi_windows[:min_size_bbi_windows_per_core]
        else: #all the main thread basic blocks
            all_core_windows[0].append(bbi_windows)
            sum_p_bbi_ser = sum_p_bbi_ser + orig_bb_probs[i]
        
        # for core in range(num_cores):
        #     print "Core ",core," :",  " :number of windows ", len(all_core_windows[core]), ", Windows ", all_core_windows[core]
        #     print "-------------------------------------------------------------------------------------"
        # raw_input("Press Enter to continue...")

    for i in range(0, len(bb_table)):
        bb_table_row = bb_table[i]
        b_t_info = bb_table_row.split(': ')
        if("OUT__" in b_t_info[2]):
            master_thread_prob.append((orig_bb_probs[i]/num_cores)/(sum_p_bbi_ser + sum_p_bbi_par/num_cores))
            other_thread_prob.append(
                (orig_bb_probs[i]/num_cores)/(sum_p_bbi_par/num_cores))
        else:
            master_thread_prob.append(
                (orig_bb_probs[i])/(sum_p_bbi_ser + sum_p_bbi_par/num_cores))
            other_thread_prob.append(0.0)

    # raw_input("Press Enter to continue...")
    for core in range (num_cores):
        core_windows_flatten = []
        for item in all_core_windows[core]:
            core_windows_flatten.extend(item)
        # print core_windows_flatten
        core_windows_flatten.sort()
        # print core_windows_flatten
        for item in core_windows_flatten:
            all_core_trace[core].extend(orig_trace[item[0]: item[1]+1])
        for ind in range(len(all_core_trace[core])):
            #Change address only if its not a shared variable and valid hex address
            if(all_core_trace[core][ind][:2] == '0x' and all_core_trace[core][ind] not in shared_variable_trace):
                all_core_trace[core][ind] = hex(int(all_core_trace[core][ind], 0) + differ*core)
        # Write Trace to File
        # o_t_file = open("Core : " + str(core) + 'trace of ' + str(num_cores) + 'cores' + '.dat', 'w')
        # for addr in all_core_trace[core]:
        #     print >>o_t_file,addr
        print "Calculating private memory reuse profile for core ", core
        if (core == 0):
            compute_reuse_profile(bb_table, all_core_trace[core], master_thread_prob, num_cores, str(core))
        else:
            compute_reuse_profile(
                bb_table, all_core_trace[core], other_thread_prob, num_cores, str(core))
            

        
    print "********* Private Traces Generated *********\n"
    
    
    # Computer other threads P(BBi)
    for distri in distributions:
        print "\n\n*/*/*/*/*/*/*   Merging all the traces to generate L3 cache trace with " + \
            distri + ' distribution'
        # First trace will be modified by ther function, so pass by value
        L3_trace = merge_traces(bb_table, all_core_trace[0][:], all_core_trace[:], num_cores, distri)
        print 'Trace Length ', len(L3_trace)
        compute_reuse_profile(bb_table, L3_trace,
                              orig_bb_probs, num_cores, distri)
        print "********* L3 Cache Shared Trace Generated ********* with " + distri + " distribution\n"
    # return L3_trace

def main(bb_file,trace_file, num_cores, bb_probs_file):
    num_cores = int(num_cores)
    bb_table = open(bb_file,'r').read().strip().split('\n')     
    bb_probs = open(bb_probs_file,'r').read().strip().split('\n')    
    bb_probs = map(lambda x: float(x), bb_probs)
    print "Basic Block Probabilities: ", bb_probs
    shared_variable_trace = []
    orig_trace = []
    if("processed_trace" in str(trace_file)):
        orig_trace = read_prep_trace_from_file(trace_file)
        print "Read from preprocessed trace"
    else:
        start = time.time()
        orig_trace = preprocess_orig_trace(bb_table,trace_file)
        end = time.time()
        print "Preprocess time ", end - start
        
    print "Generating memory trace for shared variables"
    for i in range(0,len(bb_table)):
        bb_table_row = bb_table[i]
        b_t_info = bb_table_row.split(': ')
        if("shared_trace" in b_t_info[2] or "global_var_trace" in b_t_info[2]):
            bbi_windows_w_m = get_bb_i_windows_with_mem_references(b_t_info[2], orig_trace)
            for bbi_window in bbi_windows_w_m:
                shared_variable_trace.extend(orig_trace[bbi_window[0]+1 : bbi_window[1]])
    print "Memory trace for shared variables generated"

    if(num_cores > 1):
        references = map(lambda x: int(x, 16), list(filter(lambda addr: addr[:2]=='0x', orig_trace)))
        print type(references[5])    
        #Get the maximum and minimum memory references
        print min(references), max(references)
        differ = max(references) - min(references)
        print "Address Difference ", differ
        start = time.time()
        generate_each_thread_trace_and_get_shared(
            bb_table, orig_trace, num_cores, differ, shared_variable_trace, bb_probs)
        end = time.time()
        print "Interleave time ", end - start
    else:
        # First trace will be modified by ther function, so pass by value
        compute_reuse_profile(bb_table, orig_trace,
                              bb_probs, num_cores, distributions[0])
        print "********* L3 Cache Shared Trace Generated ********* with "
    print id(merge_traces), id(orig_trace)

if __name__ == "__main__":
    #if(len(sys.argv) <= 5):   usuage()
    if(len(sys.argv) != 5):   usuage()
    else:    
        #main(sys.argv[1],sys.argv[2],sys.argv[3], sys.argv[4:])
        main(sys.argv[1],sys.argv[2], sys.argv[3], sys.argv[4])
