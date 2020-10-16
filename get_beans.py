# python3 script
# Author: Atanu Barai

import sys, csv, math
from itertools import cycle
import pandas as pd
import numpy as np



def get_reuse_prof(rp_file):
    #columns = ['sd', 'counts', 'psd']
    columns = ['sd', 'psd', 'counts']
    columns = ['sd', 'psd', 'counts']
    #columns = ['sd', 'psd']
    # reuse_prof = pd.read_csv(rp_file,delimiter='\t',names=columns)
    reuse_prof = pd.read_csv(rp_file,delimiter=',',names=columns)
    reuse_prof = reuse_prof.astype(float)
    reuse_prof = reuse_prof.sort_values(['sd'],ascending=[True])
    return reuse_prof


def get_SDbeans(sds, n_bins=5):
    sd_bins = []
    sds = list(sds)
    #bin_len = int(math.ceil((len(sds)/(n_bins*1.0))))
    bin_len = int(len(sds)/n_bins)
    # print (sds, bin_len)
    start = 0
    end = bin_len
    for i in range(n_bins):
        if i+1 is n_bins:      
            bin_sds = sds[start:]
        else:   
            bin_sds = sds[start:end]
        # print(start, end, bin_sds)
        start = end
        end = bin_len+end
        avg_sd = round(sum(bin_sds)/len(bin_sds))
        sd_bins.append(avg_sd)
    #print sd_bins
    return sd_bins



if __name__ == "__main__":         
    init = 1
    n_bins = 20
    degree = sys.argv[1]
    count_bins = []
    xs = []
    print ("SD Beans")
    for i in range(2, len(sys.argv)):
        start = sys.argv[i].find('Input_') + len('Input_')
        end = (sys.argv[i].find('/', start))
        x = sys.argv[i][start:end]
        xs.append(x)
        reuse_prof = get_reuse_prof(sys.argv[i])
        sd_bins = get_SDbeans(reuse_prof['sd'][init:],n_bins)
        count_bins.append(get_SDbeans(reuse_prof['sd'][init:],n_bins))
        print (x, degree, *sd_bins)

    print("\nCount Beans")
    for i in range(len(count_bins)):
        print(xs[i], degree, *count_bins[i])

