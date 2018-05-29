import os
import re
from collections import OrderedDict
import numpy as np
import sys

FWD_ALGO_list=[
"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
"CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
"CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
"CUDNN_CONVOLUTION_FWD_ALGO_FFT",
"CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED"]

BWD_ALGO_DATA_list= [
"CUDNN_CONVOLUTION_BWD_DATA_ALGO_0",
"CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",
"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",
"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",
"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",
"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED"]


BWD_ALGO_FILTER_list=["CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0",
"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1",
"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT",
"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3",
"CUDNN_CONVOLUTION_BWD_FILTER_WINOGRAD_NONFUSED",
"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING"]

FWD_ALGO_TENSORCORE=["CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED"
]

BWD_ALGO_DATA_TENSORCORE=["CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",
"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED"]

BWD_ALGO_FILTER_TENSORCORE=["CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1",
"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED"]

MATH_OPS_list= ['CUDNN_TENSOR_OP_MATH', 'CUDNN_DEFAULT_MATH']

def todict(LIST):
    return OrderedDict([(itm, [re.compile(itm), 0]) for itm in LIST])


def count_occurences(filepath, line_bounds, ord_dict_list, portion=0.5):
    line_lb, line_ub = line_bounds
    with open(filepath,'r') as f:
        for (num_line,line) in enumerate(f):
            if num_line > line_lb and num_line < line_ub:
                for ord_dict in ord_dict_list:
                    for key, itm in ord_dict.items():
                        if itm[0].search(line):
                            ord_dict[key][1] += 1


def rank_entries(ord_dict_list, steps):
    FWD_ALGO_TENSORCORE=["CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED"
    ]

    BWD_ALGO_DATA_TENSORCORE=["CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED"]

    BWD_ALGO_FILTER_TENSORCORE=["CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED"]

    for ord_dict in ord_dict_list:
        arr_counts = np.array([itm[1] for _, itm in ord_dict.items()])
        indices = np.argsort(arr_counts)[::-1]
        keys = list(ord_dict.keys())
        print('Trace from training step=%d to step=%d' %(steps[0], steps[1]))
        print('CUDA FUNCTION, # CUDA CALLS, TENSORCORES USAGE')
        for ind in indices:
            algo_name = keys[ind]
            if algo_name in FWD_ALGO_TENSORCORE+BWD_ALGO_DATA_TENSORCORE+BWD_ALGO_FILTER_TENSORCORE:
                tensorcore_usage = "YES"
            else:
                tensorcore_usage = "NO"
            print('%s, %d ,%s ' %(algo_name, ord_dict[algo_name][1], tensorcore_usage))
        print('\n')

def get_step_timing(logfile):
    step_1 = re.compile('step= 90')
    step_2 = re.compile('step= 100')
    times, steps = [], []
    with open(logfile, mode='r') as f:
        for line in f:
            if step_1.search(line) or step_2.search(line):
                stream = line.split(',')
                time = stream[0].split('=')[-1]
                step = stream[1].split('=')[-1]
                times.append(float(time))
                steps.append(int(step))
    return times, steps

def get_lines_bounds(times, logfile):
    pattern = re.compile('Time:')
    lines = []
    with open(logfile, mode='r') as f:
        for i,line in enumerate(f):
            if pattern.search(line):
                stream=line
                stream=line.split(' ')[-3]
                time_list = re.findall('\d+',stream)
                total_time = int(time_list[0])*3600*24 + int(time_list[1])*3600 + int(time_list[2])*60 + int(time_list[3])
                if total_time > times[0] or total_time < total_time: lines.append(i)
    return lines[0], lines[-1]

def main(argv):
    cudnn_logfile=argv[-2]
    train_logfile=argv[-1]
    # Dictionaries
    FWD_ALGO = todict(FWD_ALGO_list)
    BWD_DATA_ALGO = todict(BWD_ALGO_DATA_list)
    BWD_FILTER_ALGO = todict(BWD_ALGO_FILTER_list)
    MATH_OPS = todict(MATH_OPS_list)
    ord_dict_list = [FWD_ALGO, BWD_DATA_ALGO, BWD_FILTER_ALGO, MATH_OPS]
    # parsing
    times, steps = get_step_timing(train_logfile)
    line_lb, line_ub = get_lines_bounds(times, cudnn_logfile)
    count_occurences(cudnn_logfile, [line_lb, line_ub], ord_dict_list, portion=0.75)
    rank_entries(ord_dict_list, steps)

if __name__ == "__main__":
    main(sys.argv)
