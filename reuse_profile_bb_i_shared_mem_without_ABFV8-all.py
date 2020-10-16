"""
Author          :       Atanu Barai (atanu@nmsu.edu)
Last Modified   :       August 18, 2020
File            :       reuse_profile_bb_i_shared_mem_Ver8.py
Purpose         :       Preprocess the raw trace resulted out of modified Byfl
                        Compute the shared reuse profile of the program from the memory trace of
                        sequential execution of the program
Output          :       Write the sd,p(sd/BBi) into a file
Note            :       This version should handle fairly large trace file
"""

import sys
import re
import io
import numpy as np
import random
import time
import gc
import math
import copy
from operator import itemgetter
from collections import OrderedDict
import matplotlib.pyplot as plt

distributions = ['_uniform', '_round_robin']  # _round_robin / _uniform

# distributions = ['_round_robin']
sample_size = 10 #random number of sample


def usuage():
    print("[USUAGE]: "+__file__+" basicblocks_table processed_trace num_cores bb_probs")

def read_prep_trace_from_file(prep_trace_file):
    return open(prep_trace_file, 'r').read().strip().split('\n')

def get_bb_i_windows_fast(bbi_name, mem_trace):
    '''
    Return the window size (k,z) of a given BasicBlock
    i = start of BBi
    j = end of BBi
    '''
    bbi_wins = []
    bb_start_key = 'BB START: {}'.format(bbi_name)
    bb_end_key = 'BB DONE: {}'.format(bbi_name)
    #print bb_start_key, bb_end_key
    trace_len = len(mem_trace)
    for i in range(0, trace_len):
        if bb_start_key == mem_trace[i]:
            startIdx = i
        elif bb_end_key == mem_trace[i]:
            bbi_wins.append([startIdx, i])
    if len(bbi_wins) == 0:
        bbi_wins = [[0, 0]]
    return bbi_wins

def get_all_bb_windows_fast(bb_names, mem_trace, avoid_empty_bb=True):
    '''
    Accepts list of basic block names, memory trace and returns windows for all BBs in the list
    '''
    bb_wins = OrderedDict()
    for name in bb_names:
        bb_wins[name] = []

    startIdx = 0
    trace_len = len(mem_trace)
    interval = int(trace_len/100)
    if interval == 0:
        interval = 1
    print("Trace length: ", trace_len, " Interval: ", interval)
    if avoid_empty_bb == True:
        print('Getting BB windows from the trace, avoiding empty windows')
        for i in range(0, trace_len):
            if int(i%interval) == 0:
                print (i/interval),
                sys.stdout.flush()
            if 'BB START' in mem_trace[i]:
                startIdx = i
            elif 'BB DONE' in mem_trace[i]:
                if i != startIdx+1:
                    if list(filter(lambda addr: addr[:2] == '0x', mem_trace[startIdx:i])):
                        bb_name = mem_trace[i].split(': ')[1].strip()
                        bb_wins[bb_name].append([startIdx, i])
    elif avoid_empty_bb == False:
        print('Getting BB windows from the trace, not avoiding empty windows')
        for i in range(0, trace_len):
            if int(i%interval) == 0:
                print (i/interval),
                sys.stdout.flush()
            if 'BB START' in mem_trace[i]:
                startIdx = i
            elif 'BB DONE' in mem_trace[i]:
                bb_name = mem_trace[i].split(': ')[1]
                bb_wins[bb_name].append([startIdx, i])

    print("\nBB windows calculated")
    return bb_wins

def sample_bbi_wins(bbi_win_sizes, sample_size):
    '''
    Randomly sample 'sample_size' number of bbi_win_sizes
    '''
    sampled_wins = []
    len_bbi_win_sizes = len(bbi_win_sizes)
    if len_bbi_win_sizes > sample_size:
        indices = list(range(len_bbi_win_sizes))
        sample_indices = random.sample(indices, sample_size)
        '''
        if not len_bbi_win_sizes-1 in sample_indices:
            sample_indices[-1] = len_bbi_win_sizes-1
        '''
        for s_idx in sample_indices:
            sampled_wins.append(bbi_win_sizes[s_idx])
    else: sampled_wins = bbi_win_sizes
    return sampled_wins

def get_bbi_reuse_prof_fast(bb_i_size, orig_trace):
    sd = [0, 0]
    if bb_i_size[1] > bb_i_size[0]+1:
        bbi_trace = orig_trace[bb_i_size[0]+1:bb_i_size[1]]
        sd = []
        # Calculate the SDs for each addr in BBi
        for addr, idx in zip(bbi_trace, range(bb_i_size[0]+1, bb_i_size[1])):
            window_trace = orig_trace[:idx]
            dict_sd = {}
            addr_found = False
            for w_adx in range(0, len(window_trace)):
                w_addr = window_trace[-w_adx -1]
                if addr == w_addr:
                    addr_found = True
                    break
                if w_addr[:2] == '0x':
                    dict_sd[w_addr] = True
            if addr_found: sd.append(len(dict_sd))
            else: sd.append(-1)
    sd = np.array(sd)
    return sd

def get_prob_sd_bb_i(sd_vals):
    '''
    Return the unique reuse distances (D) and their probabilities (p(D/BBi)))
    for a given BasicBlock
    '''
    len_sd = len(sd_vals)
    uniq_sd,counts = np.unique(sd_vals, return_counts=True)
    #Compute probabilities
    p_uniq_sd_bbi = map(lambda x: x/float(len_sd), counts)
    print('Sum of p(sd/bbi) : ', np.sum(p_uniq_sd_bbi))
    return zip(uniq_sd, p_uniq_sd_bbi, counts)

def get_final_prob_bb_i(reuse_profile_bbi, bbi_prob, bbi_count):
    '''
    Multiply conditional probabilities of a given Basicblock with 'weighted probability'
    '''
    # print bbi_prob, bbi_count
    final_reuse_prof = []
    for i in range(0, len(reuse_profile_bbi)):
        reuse_profile_bbi[i][1] = float(bbi_prob) * float(reuse_profile_bbi[i][1])
        reuse_profile_bbi[i][2] = float(bbi_count) * float(reuse_profile_bbi[i][2])
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
    print("********** Done adding the duplicate probabilities ***********")
    return res

def compute_reuse_profile(bb_names, target_mem_trace, bb_probs, bb_counts, count_factor, num_cores,\
    distri):
    print("\n\n***************** Computing Reuse Profile of the given trace *****************")
    final_reuse_profile = []
    final_reuse_prof_serial_sec = []
    final_reuse_prof_parallel_sec = []
    sum_pbbs = 0
    all_bb_windows = get_all_bb_windows_fast(bb_names, target_mem_trace)
    for i in range(0, len(bb_names)):
        print("\n------------> Processing BasicBlock ", bb_names[i], ", Probability : ", bb_probs[i])

        if bb_probs[i] == 0.0:
            print("BB with probability 0.0., skipping")
            continue
        sum_pbbs += bb_probs[i]
        bbi_windows = all_bb_windows[bb_names[i]]
        if len(bbi_windows) == 0:
            print("Blank BB, skipping")
            continue
        bbi_win_sizes = sample_bbi_wins(bbi_windows, sample_size)
        res_sds = np.array([])
        for bb_i_size in bbi_win_sizes:
            res_sds = np.concatenate([res_sds, get_bbi_reuse_prof_fast(bb_i_size, target_mem_trace)])
        zipped_sd_psd_i = get_prob_sd_bb_i(res_sds)
        res_sd_psd_i = [list(sd_psd) for sd_psd in zipped_sd_psd_i]
        f_res_sd_psd_i = []
        if "OUT__" in bb_names[i]:
            f_res_sd_psd_i = get_final_prob_bb_i(res_sd_psd_i, bb_probs[i], bb_counts[i]/(len(bbi_win_sizes)*count_factor))
        else:
            f_res_sd_psd_i = get_final_prob_bb_i(res_sd_psd_i, bb_probs[i], bb_counts[i]/len(bbi_win_sizes))
        for f_sd_psd_i in f_res_sd_psd_i:  #TODO: Only seperate parallel, serial in case of L3 trace
            final_reuse_profile.append(f_sd_psd_i)
            if distri in distributions:
                if "OUT__" in bb_names[i]:
                    final_reuse_prof_parallel_sec.append(f_sd_psd_i)
                else:
                    final_reuse_prof_serial_sec.append(f_sd_psd_i)
        print("Stack Distance Stats of BasicBlock ", bb_names[i])
        print("mean %f min %f max %f std %f" %(np.mean(res_sds), np.min(res_sds), np.max(res_sds), np.std(res_sds)))
    #Get the final reuse profile for the program by merging the probabilities for all the BBi
    #print final_reuse_profile
    all_bbi_sd_profile = get_all_bbi_reuse_profile(final_reuse_profile)
    # crit_bbi_sd_profile = merge_reuse_profile(final_reuse_prof_crit_sec)
    if distri in distributions:
        parallel_bbi_sd_profile = get_all_bbi_reuse_profile(
            final_reuse_prof_parallel_sec)
        serial_bbi_sd_profile = get_all_bbi_reuse_profile(
            final_reuse_prof_serial_sec)
    print("\nSize of the final reuse profile : ", len(all_bbi_sd_profile))

    sum_f = sum(value[0] for key, value in all_bbi_sd_profile.iteritems())
    #stack_dist = []
    #probability_sd = []
    with open("reuse_profile_mimic_"+str(num_cores) + "_cores" + distri + ".dat", "w") as f:
        for key, value in all_bbi_sd_profile.iteritems():
            #stack_dist.append(key)
            #probability_sd.append(value[0])
            f.write(str(key).ljust(4)+" , "+str(value[0]).ljust(20)+" , "+str(int(round(value[1]))).ljust(4)+"\n")
            #f.write(str(key).ljust(4)+" , "+str(value[0]).ljust(20)+"\n")
    # plot_graph(stack_dist, probability_sd, num_cores, distri)
    # with open("reuse_profile_crit_sec_"+ distri +".dat","w") as f:
    #     for key,value in crit_bbi_sd_profile.iteritems():
    #         f.write(str(key).ljust(4)+" , "+str(value[0]).ljust(20)+" , "+str(value[1]).ljust(4)+"\n")
    #         #f.write(str(key).ljust(4)+" , "+str(value[0]).ljust(20)+"\n")

    if distri in distributions:
        with open("reuse_profile_parallel_sec_"+str(num_cores)+ "_cores" + distri + ".dat", "w") as f:
            for key,value in parallel_bbi_sd_profile.iteritems():
                f.write(str(key).ljust(4)+" , "+str(value[0]).ljust(20)+" , "+str(int(round(value[1]))).ljust(4)+"\n")
                #f.write(str(key).ljust(4)+" , "+str(value[0]).ljust(20)+"\n")

        with open("reuse_profile_serial_sec_"+str(num_cores)+ "_cores" + distri + ".dat", "w") as f:
            for key,value in serial_bbi_sd_profile.iteritems():
                f.write(str(key).ljust(4)+" , "+str(value[0]).ljust(20)+" , "+str(int(round(value[1]))).ljust(4)+"\n")
                #f.write(str(key).ljust(4)+" , "+str(value[0]).ljust(20)+"\n")

    print("Finished preparing the final reuse_profile of the program")
    print("Sum of p(BBi) : ", sum_pbbs)
    print("Final sum of the reuse profile probabilities", sum_f)
    # return all_bbi_sd_profile

def interleave_traces_fast(bb_names, all_core_traces, num_cores, distri):
    ## Interleaves the traces of all cores according to distribution
    interleaved_trace = []
    all_core_bb_wins = []
    #Get BB windows from all core's trace but core0
    for core in range(1, num_cores):
        print("Get BB windows of all BBs in trace of core", core)
        all_core_bb_wins.append(get_all_bb_windows_fast(bb_names, all_core_traces[core], False))
    
    print(len(all_core_bb_wins))
    # Now merge the traces following execution order from core0's trace
    print("Now interleaving the traces")
    startIdx = 0
    trace_len = len(all_core_traces[0])
    interval = int(trace_len/100)
    if interval == 0:
        interval = 1
    print("Trace length: ", trace_len, " Interval: ", interval)
    i = 0
    # trace_all_cores_per_bb_occurance = [[]for _ in range(num_cores)]
    
    while (i < trace_len):
        if int(i%interval) == 0:
            print (i/interval),
            sys.stdout.flush()
        entry_i = all_core_traces[0][i]
        # print i, entry_i
        if 'OUT__' in entry_i:
            if 'BB START' in entry_i:
                interleaved_trace.append(entry_i)
                bb_i_name = entry_i.split(': ')[1]
                i += 1
                startIdx = i
                while ('BB DONE' not in all_core_traces[0][i]):
                    if int(i%interval) == 0:
                        print (i/interval),
                        sys.stdout.flush()
                    # trace_all_cores_per_bb_occurance[0].append(all_core_traces[0][i])
                    i += 1
                interleaved_trace_per_bb_iter = []
                interleaved_trace_per_bb_iter.extend(all_core_traces[0][startIdx:i])
                for core in range(1, num_cores):
                    try:
                        window = all_core_bb_wins[core - 1][bb_i_name].pop(0)
                        interleaved_trace_per_bb_iter.extend(all_core_traces[core][(window[0]+1):window[1]])
                    except:
                        pass
                if distri == '_uniform':
                    random.shuffle(interleaved_trace_per_bb_iter)
                interleaved_trace.extend(interleaved_trace_per_bb_iter)
            else:
                interleaved_trace.append(entry_i)
                i += 1
        else:
            interleaved_trace.append(entry_i)
            i += 1

    o_t_file = open('New_Interleaved_L3_trace_' + str(num_cores) + '_cores_' + distri +'.dat','w')
    for addr in interleaved_trace:
       print >>o_t_file,addr
    return interleaved_trace

def generate_each_thread_trace_and_get_shared(bb_names, orig_trace, num_cores, differ, shared_variable_trace, \
    orig_bb_probs, orig_bb_counts):
    print("Generating trace for each core")
    # shared_variable_trace = ['0x000000000000']    
    
    all_core_trace = [[]for _ in range(num_cores)]
    all_core_windows = [[]for _ in range(num_cores)]
    sum_p_bbi_ser = 0.0
    sum_p_bbi_par = 0.0
    master_thread_prob = []
    other_thread_prob = []
    master_thread_bbi_win_count = []
    other_thread_bbi_win_count = []
    # all_core_pBB = [[]for _ in range(num_cores)]
    all_bb_windows = get_all_bb_windows_fast(bb_names,orig_trace,avoid_empty_bb=False)
    for i in range(0, len(bb_names)):
        print("\n\n-----------------------------------------------------------")
        print(bb_names[i])
        print("Calculating BBi Windows")
        bbi_windows = all_bb_windows[bb_names[i]]
        if len(bbi_windows) == 0:
            print("This BB does not appear on the trace")
            #TODO: Edit here
            master_thread_bbi_win_count.append(0.0)
            other_thread_bbi_win_count.append(0.0)
            continue
        # if "_ZL14OUT__2__6363__Pv, for.body3" in b_t_info[2]:
        #     print bbi_windows
        len_bbi_win_sizes = len(bbi_windows)
        print("LENGTH of basic block windows" ,len_bbi_win_sizes)
        # if len_bbi_win_sizes == 1:
        #     print bbi_windows
        # print " BBi Windows: ", bbi_windows
        min_size_bbi_windows_per_core = 1
        max_size_bbi_windows_per_core = 1
        remaining_windows = 0

        if("OUT__" in bb_names[i]):
            sum_p_bbi_par = sum_p_bbi_par + orig_bb_probs[i]
            if(len_bbi_win_sizes < num_cores):
                min_size_bbi_windows_per_core = 1
                master_thread_bbi_win_count.append(1.0)
                other_thread_bbi_win_count.append(1.0)

            else:
                # The early threads will execute more loop iterations if number of iterations can not be evenly \
                # distributed
                min_size_bbi_windows_per_core = int(math.floor(
                    float(len_bbi_win_sizes) / float(num_cores)))
                max_size_bbi_windows_per_core = int(
                    math.ceil(float(len_bbi_win_sizes) / float(num_cores)))
                print("MAX ", max_size_bbi_windows_per_core, " MIN ", min_size_bbi_windows_per_core)
                remaining_windows = len_bbi_win_sizes % num_cores
                master_thread_bbi_win_count.append(max_size_bbi_windows_per_core)
                other_thread_bbi_win_count.append(min_size_bbi_windows_per_core)

            if(len_bbi_win_sizes == 1):
                for core in range (0, num_cores):
                    all_core_windows[core].append(bbi_windows)
            else:
                max_core = len_bbi_win_sizes if len_bbi_win_sizes < num_cores else num_cores
                for core in range (0, max_core):
                    if(core < remaining_windows):
                        print("Core ", core, " In MAX")
                        #default openmp scheduler
                        all_core_windows[core].append(bbi_windows[:max_size_bbi_windows_per_core]) 
                        del bbi_windows[:max_size_bbi_windows_per_core]                        
                    else:
                        print("Core ", core, " In MIN")
                        #default openmp scheduler
                        all_core_windows[core].append(bbi_windows[:min_size_bbi_windows_per_core])
                        del bbi_windows[:min_size_bbi_windows_per_core]
        else: #all the main thread basic blocks
            all_core_windows[0].append(bbi_windows)
            sum_p_bbi_ser = sum_p_bbi_ser + orig_bb_probs[i]
            master_thread_bbi_win_count.append(orig_bb_counts[i])
            other_thread_bbi_win_count.append(0.0)

        # for core in range(num_cores):
        #     print "Core ",core," :",  " :number of windows ", len(all_core_windows[core]), ", Windows ", all_core_windows[core]
        #     print "-------------------------------------------------------------------------------------"
        # raw_input("Press Enter to continue...")

    for i in range(0, len(bb_names)):
        if("OUT__" in bb_names[i]):
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
        # Sort the windows belonging to a core so that the trace holds order of execution
        core_windows_flatten.sort()
        # Get memory trace for different cores from their windows
        for item in core_windows_flatten:
            all_core_trace[core].extend(orig_trace[item[0]: item[1]+1])
        # Add offset to traces of all cores but core0
        for ind in range(len(all_core_trace[core])):
            #Change address only if its not a shared variable and valid hex address
            if(all_core_trace[core][ind][:2] == '0x' and all_core_trace[core][ind] not in shared_variable_trace):
                all_core_trace[core][ind] = hex(int(all_core_trace[core][ind], 0) + differ*core)
        
        # Write Trace to File
        o_t_file = open("Core : " + str(core) + 'trace of ' + str(num_cores) + 'cores' + '.dat', 'w')
        for addr in all_core_trace[core]:
            print >>o_t_file,addr
        
        # print "Calculating private memory reuse profile for core ", core
        # if (core == 0):
        #     print "Computing Reuse Profile for core 0 / master thread"
        #     compute_reuse_profile(
        #         bb_names, all_core_trace[core], master_thread_prob, master_thread_bbi_win_count, 1, num_cores, str(core))
        # elif (core == 1):
        #     print "Computing Reuse Profile for core 1"
        #     compute_reuse_profile(
        #         bb_names, all_core_trace[core], other_thread_prob, other_thread_bbi_win_count, 1, num_cores, str(core))
        
    print("********* Private Traces Generated *********\n")
    # Computer other threads P(BBi)
    for distri in distributions:
        print("\n\n*/*/*/*/*/*/*   Merging all the traces to generate L3 cache trace with " + \
            distri + ' distribution')
        # First trace will be modified by ther function, so pass by value
        startTime = time.time()
        L3_trace = interleave_traces_fast(bb_names, all_core_trace, num_cores, distri)
        # print '\nTrace Length ', len(L3_trace)
        endTime = time.time()
        
        # compute_reuse_profile(bb_names, L3_trace,
        #                       orig_bb_probs, orig_bb_counts, num_cores , num_cores, distri)
        print("********* L3 Cache Shared Reuse Profile Generated ********* with " + distri + " distribution\n")
        
    # return L3_trace

def main(bb_file, trace_file, num_cores, bb_counts_file):
    num_cores = int(num_cores)
    # Get BB names from BB info file generated by Byfl
    bb_names = [row.split(': ')[2] for row in open(bb_file,'r').read().strip().split('\n')]
    # Get BB counts table from BB counts file
    bb_counts_table = [row.split(': ') for row in open(bb_counts_file,'r').read().strip().split('\n')]
    '''
    INFO: Sorting BB counts is not needed as OrderedDict is being used to list the BB counts
    '''
    orig_bb_counts = [float(item[1]) for item in bb_counts_table]
    sum_bb_counts = sum(orig_bb_counts)
    orig_bb_probs = list(map(lambda x: x/sum_bb_counts, orig_bb_counts))
    shared_variable_trace = []
    orig_trace = []
    if("processed_" in str(trace_file)):
        orig_trace = read_prep_trace_from_file(trace_file)
        print("Read from preprocessed trace")
    else:
        print("Enter a processed trace. Raw trace can be processed using bb_overlap_checker.py")
    start = time.time()
    all_bb_windows = get_all_bb_windows_fast(bb_names, orig_trace, False)
    end = time.time()
    print("BB windows of original memory trace calculation time ", end - start)
    
    print("START: Memory trace for shared variables")
    for bb_i_name in bb_names:
        if("shared_trace" in bb_i_name or "global_var_trace" in bb_i_name):
            bbi_windows_w_m = all_bb_windows[bb_i_name]
            for bbi_window in bbi_windows_w_m:
                shared_variable_trace.extend(orig_trace[bbi_window[0]+1 : bbi_window[1]])
    print("DONE: Memory trace for shared variables")
    if(num_cores > 1):
        references = map(lambda x: int(x, 16), list(filter(lambda addr: addr[:2]=='0x', orig_trace)))
        print(type(references[5]))
        #Get the maximum and minimum memory references
        print(min(references), max(references))
        differ = max(references) - min(references)
        print("Address Difference ", differ)
        start = time.time()
        generate_each_thread_trace_and_get_shared(
            bb_names, orig_trace, num_cores, differ, shared_variable_trace, orig_bb_probs, orig_bb_counts)
        end = time.time()
        print("Interleave time ", end - start)
    else:
        # First trace will be modified by ther function, so pass by value
        compute_reuse_profile(bb_names, orig_trace,
                              orig_bb_probs, orig_bb_counts, 1 , num_cores, distributions[0])
    

if __name__ == "__main__":
    #if(len(sys.argv) <= 5):   usuage()
    if(len(sys.argv) != 5):   usuage()
    else:    
        #main(sys.argv[1],sys.argv[2],sys.argv[3], sys.argv[4:])
        main(sys.argv[1],sys.argv[2], sys.argv[3], sys.argv[4])
