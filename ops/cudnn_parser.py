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


def count_occurences(filepath, ord_dict_list, portion=0.5):
    num_lines = 0
    with open(filepath,'r') as f:
            for line in f:
                num_lines += 1
    cutoff = int(num_lines*3/4)
    with open(filepath,'r') as f:
        for num_line, line in enumerate(f):
            if num_line > cutoff:
                for ord_dict in ord_dict_list:
                    for key, itm in ord_dict.items():
                        if itm[0].search(line):
                            ord_dict[key][1] += 1

def rank_entries(ord_dict_list):
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
        print('CUDA FUNCTION, # CUDA CALLS, TENSORCORES USAGE')
        for ind in indices:
            algo_name = keys[ind]
            if algo_name in FWD_ALGO_TENSORCORE+BWD_ALGO_DATA_TENSORCORE+BWD_ALGO_FILTER_TENSORCORE:
                tensorcore_usage = "YES"
            else:
                tensorcore_usage = "NO"
            print('%s, %d ,%s ' %(algo_name, ord_dict[algo_name][1], tensorcore_usage))
        print('\n')

def main(argv):
    file_path=argv[-1]
    FWD_ALGO = todict(FWD_ALGO_list)
    BWD_DATA_ALGO = todict(BWD_ALGO_DATA_list)
    BWD_FILTER_ALGO = todict(BWD_ALGO_FILTER_list)
    MATH_OPS = todict(MATH_OPS_list)

    ord_dict_list = [FWD_ALGO, BWD_DATA_ALGO, BWD_FILTER_ALGO, MATH_OPS]
    count_occurences(file_path, ord_dict_list, portion=0.75)

    rank_entries(ord_dict_list)

if __name__ == "__main__":
    main(sys.argv)
