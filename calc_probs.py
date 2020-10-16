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

import sys
import re
import io
import cPickle as cp
import numpy as np
import random
import time
import gc
import math
import copy
from operator import itemgetter
from collections import OrderedDict
import matplotlib.pyplot as plt


def usuage():
    print "[USUAGE]: "+__file__+" Basicblocks_table trace"


def preprocess_orig_trace(bb_table, trace_file):
    '''
    Preprocess original trace to include BasicBlock names and their END
    '''
    orig_mem_trace = open(trace_file, 'r').read().strip()
    # Fixing the irregular start of the basic blocks, meaning there are >=two consecutive BB starts before it ends
    start_idx = [match.start()
                 for match in re.finditer('BB START: ', orig_mem_trace)]
    for i in range(0, len(start_idx)-1):
        s_id = start_idx[i]
        nl_id = orig_mem_trace[s_id:].find('\n')
        rand_id = s_id+nl_id+1
        #bb_rand_name = orig_mem_trace[s_id:rand_id]
        if start_idx[i+1] == rand_id:
            #print(i, s_id, start_idx[i+1])
            rand_str = orig_mem_trace[s_id+len('BB START: '):rand_id]
            bb_done = 'BB DONE: '+rand_str
            orig_mem_trace = orig_mem_trace[:rand_id] + \
                bb_done+orig_mem_trace[rand_id:]

            def add_soffset(_a):
                return _a+len(bb_done)
            start_idx[i:] = map(add_soffset, start_idx[i:])
    print('Length of original trace after BB START Prep {}'.format(
        len(orig_mem_trace)))
    # Fixing the irregular end of the basic blocks, meaning there are >=two consecutive BB ends
    end_idx = [match.start()
               for match in re.finditer('BB DONE: ', orig_mem_trace)]
    for i in range(0, len(end_idx)-1):
        curr_eid = end_idx[i]
        next_eid = end_idx[i+1]
        eid_line = curr_eid+orig_mem_trace[curr_eid:].find('\n')+1
        next_start_id = orig_mem_trace[curr_eid:next_eid].find('BB START: ')
        if next_start_id == -1:
            rand_id = next_eid+orig_mem_trace[next_eid:].find('\n')+1
            rand_str = orig_mem_trace[next_eid+len('BB DONE: '):rand_id]
            bb_start = 'BB START: '+rand_str
            orig_mem_trace = orig_mem_trace[:eid_line] + \
                bb_start+orig_mem_trace[eid_line:]

            def add_eoffset(_a):
                return _a+len(bb_start)
            end_idx[i:] = map(add_eoffset, end_idx[i:])
    print('Length of original trace after BB DONE Prep {}'.format(len(orig_mem_trace)))
    #print bb_table
    for b_t in bb_table:
        b_t_info = b_t.split(': ')
        #print b_t_info[0]+":", b_t_info[2]
        if b_t_info[1] in orig_mem_trace:
            orig_mem_trace = orig_mem_trace.replace(b_t_info[1], b_t_info[2])

            orig_mem_trace = orig_mem_trace.replace('LOAD: ', '')
            orig_mem_trace = orig_mem_trace.replace('STORE: ', '')

    orig_mem_trace = orig_mem_trace.split('\n')
    o_t_file = open('processed_trace.dat', 'w')
    for addr in orig_mem_trace:
        print >>o_t_file, addr
    print "************** Finished preprocessing the trace *************"
    return orig_mem_trace
    # return orig_mem_trace


def read_prep_trace_from_file(prep_trace_file):
    return open(prep_trace_file, 'r').read().strip().split('\n')


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
        for i, x in enumerate(orig_trace):
            if x == bb_start_key:
                j = orig_trace[i:].index(bb_end_key)+i
                if j != i+1:
                    bbi_wins.append([i, j])
                    # if list(filter(lambda addr: addr[:2] == '0x', orig_trace[i:j])):
                    #     bbi_wins.append([i,j])

    return bbi_wins


def isListEmpty(inList):
    if isinstance(inList, list):  # Is a list
        return all(map(isListEmpty, inList))
    return False  # Not a list


def is_blank_bb(bbi_windows, orig_trace):
    '''
    Check if any basicblock doesn't have any memory trace between bb_start and bb_end
    '''
    if(bbi_windows == [[0, 0]] or len(bbi_windows) == 0):
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


def main(bb_file, trace_file):
    bb_table = open(bb_file, 'r').read().strip().split('\n')
    sum_bb_counts = 0
    bb_probs = []
    orig_trace = []
    if("processed_trace" in str(trace_file)):
        orig_trace = read_prep_trace_from_file(trace_file)
        print "Read from preprocessed trace"
    else:
        start = time.time()
        orig_trace = preprocess_orig_trace(bb_table, trace_file)
        end = time.time()
        print "Preprocess time ", end - start

    print "Calculating BBi probabilities START"
    for i in range(0, len(bb_table)):
        bb_table_row = bb_table[i]
        b_t_info = bb_table_row.split(': ')
        bbi_windows_w_m = get_bb_i_windows_with_mem_references(
            b_t_info[2], orig_trace)
        bb_probs.append(float(len(bbi_windows_w_m)))
        sum_bb_counts += float(len(bbi_windows_w_m))
    bb_probs = list(map(lambda x: x/sum_bb_counts, bb_probs))
    o_t_file = open("bb_probabilities.dat", 'w')
    for prob in bb_probs:
        print >>o_t_file, prob
    print "Calculating BBi probabilities END"


if __name__ == "__main__":
    #if(len(sys.argv) <= 5):   usuage()
    if(len(sys.argv) != 3):
        usuage()
    else:
        #main(sys.argv[1],sys.argv[2],sys.argv[3], sys.argv[4:])
        main(sys.argv[1], sys.argv[2])
