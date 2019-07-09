import os
import math
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


def count_occurences(filepath, line_bounds, ord_dict_list):
    line_lb, line_ub = line_bounds
    with open(filepath,'r') as f:
        for (num_line,line) in enumerate(f):
            if num_line > line_lb and num_line < line_ub:
                for ord_dict in ord_dict_list:
                    for key, itm in ord_dict.items():
                        if itm[0].search(line):
                            ord_dict[key][1] += 1

def ceil_lg2(n):
    return math.ceil(math.log(n) / math.log(2))

def extract_conv_ops(filepath, line_bounds, steps, times, debug=False):
    line_lb, line_ub = line_bounds
    total_flops = 0.0
    flops = 0.0
    dims = []
    num_line = 0
    with open(filepath, 'r') as f:
        for _ in range(line_lb):
            f.readline()
            num_line += 1
        lines = f.__iter__()
        l = lines.readline()
        num_line += 1

        while 'xfunction cudnnConvolutionBackward' not in l: 
            if 'function cudnnConvolutionForward' not in l:
            #if 'function cudnnConvolutionBackwardFilter' not in l:
                try:
                    l = lines.readline()
                    num_line += 1
                except StopIteration:
                    break
                continue
            dims = []
            for i in range(3):
                while 'dimA' not in l:
                    l = lines.readline()
                    num_line += 1
                m = re.search(r'dimA: type=int; val=\[(\d+),(\d+),(\d+),(\d+)\]', l)
                assert m is not None
                dims.append(tuple(int(d) for d in m.groups()))
                l = lines.readline()
                num_line += 1
            while 'dataType' not in l:
                l = lines.readline()
                num_line += 1
            dtype = re.search(r'val=CUDNN_DATA_(\S+)', l).group(1)
            while 'mathType' not in l:
                l = lines.readline()
                num_line += 1
            mtype = re.search(r'val=CUDNN_(\S+)_MATH', l).group(1)
            while 'algo' not in l:
                l = lines.readline()
                num_line += 1
            #algo = re.search(r'val=CUDNN_CONVOLUTION_FWD_ALGO_(\S+)', l).group(1)
            algo = re.search(r'val=CUDNN_CONVOLUTION_(\S+)', l).group(1)
            t_in, t_filt, t_out = dims
            if 'ALGO_FFT' in l:
                # fft size needs to be sum of input and filter dimensions to
                #  allow for zero padding
                fft_h = t_in[2] + t_filt[2]
                fft_w = t_in[3] + t_filt[3]
                fft_flops = (5 * fft_h * fft_w *
                             ceil_lg2(fft_h) * ceil_lg2(fft_w))
                # we do NC + KC forward ffts and NK backwards ffts
                num_ffts = ((t_in[0] * t_in[1]) +
                            (t_filt[0] * t_filt[1]) +
                            (t_out[0] * t_out[1]))
                # and finally we need NKC element-wise products in frequency space
                freq_mults = (t_in[0] * t_filt[0] * t_filt[1] *
                              fft_h * fft_w)
                flops = fft_flops * num_ffts + freq_mults
            else:
                flops = (2 *
                         t_out[0] *  # N
                         t_out[2] *  # H
                         t_out[3] *  # W
                         t_in[1] *   # C
                         t_out[1] *  # K
                         t_filt[2] * # R
                         t_filt[3])  # S
            if num_line >= line_lb and num_line <= line_ub:
                total_flops += flops
                if debug:
                    print('in={} filt={} out={} flops={} algo={} dtype={} mtype={}'.format(t_in, t_filt, t_out, flops, algo, dtype, mtype))
            if num_line > line_ub:
                break
    print('Trace from training step=%d to step=%d' %(steps[0], steps[1]))
    print('Training Step = %2.3e FLOP (floating-point operations)' % (3*total_flops/(steps[-1] - steps[0])))
    if debug:
        print('inference = %2.2e FLOP' % (total_flops/(steps[-1] - steps[0])))
    print('Training Step = %2.3e FLOPS (floating-point operations per sec)\n' % (3*total_flops/(times[-1] - times[0])))
    if debug:
        print('inference = %2.2e FLOPS' % (total_flops/(times[-1] - times[0])))
    return total_flops

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

def get_step_timing(logfile, step_start=90, step_end=100):
    step_1 = re.compile('step= %d' %step_start)
    step_2 = re.compile('step= %d' %step_end)
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
    cudnn_times = []
    #print(times)
    with open(logfile, mode='r') as f:
        for i,line in enumerate(f):
            if pattern.search(line):
                time_list = re.findall('\d+',line)
                if len(time_list) == 11:
                    # assert len(time_list) == 11, print('Time format is not as expected in Line %d: Found %s, Expected: len(Time)=11. Results may be wrong.'
                    # % (i, format(time_list)))
                    hour,minute,sec,millsec = time_list[3:7]
                    total_time = int(hour) * 3600  
                    total_time += int(minute) * 60
                    total_time += int(sec)
                    total_time += int(millsec) * 10 ** (-len(millsec))
                    cudnn_times.append(total_time)
                if len(cudnn_times) > 1:
                    total_time -= cudnn_times[0]  # assume that first printed step lines up with cudnn start of trace
                    if total_time >= times[0] and total_time <= times[1]: 
                        lines.append(i)
    return lines[0], lines[-1]

def main(argv):
    if len(argv) == 1:
        print('Usage: python cudnn_parser.py cudnn_logfile train_logfile step_start step_end.')
    else:
        cudnn_logfile, train_logfile, step_start, step_end = argv[1:]
    # Dictionaries
    FWD_ALGO = todict(FWD_ALGO_list)
    BWD_DATA_ALGO = todict(BWD_ALGO_DATA_list)
    BWD_FILTER_ALGO = todict(BWD_ALGO_FILTER_list)
    MATH_OPS = todict(MATH_OPS_list)
    ord_dict_list = [FWD_ALGO, BWD_DATA_ALGO, BWD_FILTER_ALGO, MATH_OPS]
    # parsing
    times, steps = get_step_timing(train_logfile,step_start=int(step_start), step_end=int(step_end))
    print(times, steps)
    line_lb, line_ub = get_lines_bounds(times, cudnn_logfile)
    extract_conv_ops(cudnn_logfile, [line_lb, line_ub], steps, times)
    count_occurences(cudnn_logfile, [line_lb, line_ub], ord_dict_list)
    rank_entries(ord_dict_list, steps)

if __name__ == "__main__":
    main(sys.argv)
