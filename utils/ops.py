"""
Created on 03/27/18.
@author: Numan Laanait, Suhas Somnath
"""
import json
import numpy
from collections import OrderedDict
import joblib
import sys
from os import listdir, path
import re
import matplotlib.pyplot as plt
import matplotlib as mpl


def json_to_ordered_dict(file):
    """
    Reads a timeline.json file output by Tensorflow/libcupti and returns and OrderedDict object
    :param file: .json file.
    :return: OrderedDict
    """
    with open(file, mode='r') as f:
        def _as_ordered_dict(val):
            return OrderedDict(val)

        output = json.load(f, object_hook=_as_ordered_dict, object_pairs_hook=_as_ordered_dict)
        dic = OrderedDict(output)

    return dic


def get_all_ops(trace_dic):
    """
    Params:
    trace_dic: collections.OrderedDict of traceEvent
    Return: list of dictionaries of all ops.
    """
    try:
        trace_events = trace_dic['traceEvents']
    except KeyError:
        print('Not valid GPU trace dict object.')
        sys.exit()
    all_ops = []
    for trace in trace_events:
        try:
            if trace['cat'] == 'Op':
                all_ops.append(trace)
        except KeyError:
            pass
    return all_ops


def get_stream_all(trace_dic):
    """
    Params:
    trace_dic: collections.OrderedDict of traceEvent
    Return: pid of GPU/stream:all, (stream, pid) dictionary
    """
    try:
        trace_events = trace_dic['traceEvents']
    except KeyError:
        print('Not valid GPU trace dict object.')
        sys.exit()
    all_procs = []
    for trace in trace_events:
        try:
            if trace['name'] == 'process_name':
                all_procs.append((trace['args']['name'], trace['pid']))
        except KeyError:
            pass
    dic_procs = dict(all_procs)
    pid = dic_procs['/device:GPU:0/stream:all Compute']
    return dic_procs, pid


def get_unique_ops_names(all_ops):
    """
    Find unique op names.
    Params:
    all_ops: list, of dictionary of all operations.
    Return: list of unique op names.
    """
    return set(op['name'] for op in all_ops)


def get_wall_duration(op_names, all_ops, pid_list=(11, 7, 13, 15, 9)):
    """
    Calculates wall duration for each op in op_names.
    Params:
    op_names: list (str), names of ops of interest.
    pid_list: list (str), names of pid to include.
    all_ops: output of get_all_ops().
    Return:
    total wall duration, dict['op'] = wall duration.
    """
    # 1. Construct dictionary of op with name matching op_names
    ops_dic = OrderedDict()
    for name in op_names:
        ops = []
        for op in all_ops:
            if op['name'] == name:
                ops.append(op)
        ops_dic[name] = ops

    # 2. get duration for each op
    op_dict = OrderedDict()
    total_dur = 0
    for op_name in op_names:
        op_dur = 0
        for itm in ops_dic[op_name]:
            if itm['pid'] in pid_list:
                op_dur += itm['dur']
        op_dict[op_name] = op_dur * 1e-3  # convert from us to ms
        total_dur += op_dur * 1e-3

    # fixing the NCCL key:
    op_dict['unknown (nccl AllReduceKernel_sum_)'] = op_dict.pop('unknown')

    # Sorting durations:
    sorted_dur = sorted(op_dict.items(), key=lambda x: x[1])[::-1]
    # sorted_dur = sorted(op_dict.items(), key=operator.itemgetter(1))

    return OrderedDict(sorted_dur), total_dur


def print_timeline_stats(sorted_dur, total_dur, min_msec=5):
    """
    Prints the total time and times per op so long as the time was > min_msec
    :param sorted_dur: OrderedDict object with time per op. Times in msec
    :param total_dur: Number - total wall time per step. Time in msec
    :param min_msec: Number, optional - minimum wall time for op
    """
    print('Total Wall Duration (ms): %4.3f\n' % total_dur)
    print('OPS with wall duration > 5 ms:')
    for key, val in sorted_dur.items():
        if val > min_msec:
            print('%s : %3.3f ms' % (key, val))


def parse_single_timeline(curr_file):
    """
    Parses a single timeline file and extracts the time per op and total wall time

    :param curr_file: str / unicode - path to a single timeline .json file
    :return dicts: OrderedDict object with time per op. Times in msec
    :return tot_times: Number - total wall time per step. Time in msec
    """
    dic = json_to_ordered_dict(curr_file)
    all_ops = get_all_ops(dic)
    unique_op_names = get_unique_ops_names(all_ops)
    proc_dic, stream_all_pid = get_stream_all(dic)
    sorted_dur_dicts, total_dur = get_wall_duration(unique_op_names, all_ops, pid_list=[stream_all_pid])
    return sorted_dur_dicts, total_dur


def parse_all_timeline_files(folder, prefix='timeline', suffix='.json'):
    """
    Parses all timeline files in the given dictionary to extract the times per op and total wall time

    :param folder: str / unicode - path to directory containing all the timeline json files
    :param prefix: str / unicode (optional) - Prefix for the file names. Default = 'timeline'
    :param suffix: str / unicode (optional) - suffix for the file names. Default = '.json'
    :return dicts: list of OrderedDict objects per timeline file. Times in msec
    :return tot_times: list of Numbers with the total wall time per step. Times in msec
    """

    files = []
    for name in listdir(folder):
        if name.startswith(prefix) and name.endswith(suffix):
            files.append(path.join(path.abspath(folder), name))

    dicts = []
    tot_times = []
    if len(files) > 16:
        cores = 4
        values = [joblib.delayed(parse_single_timeline)(curr_file) for curr_file in files]
        results = joblib.Parallel(n_jobs=cores)(values)
        for item in results:
            dicts.append(item[0])
            tot_times.append(item[1])
    else:
        for curr_file in files:
            sorted_dur_dicts, total_dur = parse_single_timeline(curr_file)
            dicts.append(sorted_dur_dicts)
            tot_times.append(total_dur)

    return dicts, tot_times


def visualize_op_times(dicts, tot_times, do_hists=True, nrows=3, ncols=3):
    """
    Plots the total time, time taken by the N-1 most time consuming ops

    :param dicts: list of OrderedDict objects per timeline file. Times in msec
    :param tot_times: list of Numbers with the total wall time per step. Times in msec
    :param do_hists: bool, optional - if True - plots histograms of the times. Else, bar graph
    :param nrows: int, optional - Number of rows in the plot
    :param ncols: int, optional - Number of columns in the plot
    :return: fig: matplotlib.pyplot.Figure object
    """
    mpl.rc('figure', figsize=(5, 5))
    mpl.rc('lines', linewidth=2)
    mpl.rc('axes', labelsize=16, titlesize=16)
    mpl.rc('figure', titlesize=20)
    mpl.rc('font', size=14)  # global font size
    mpl.rc('legend', fontsize=16, fancybox=True)
    mpl.rc('xtick.major', size=6)
    mpl.rc('xtick.minor', size=4)

    if do_hists:
        y_label = 'Counts'
        x_label = 'Time (msec)'
    else:
        y_label = 'Time (msec)'
        x_label = 'Horovod Rank'

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.75 * nrows, 3 * ncols))
    if do_hists:
        axes.flat[0].hist(tot_times)
    else:
        axes.flat[0].bar(range(len(tot_times)), tot_times)
    axes.flat[0].set_title('Step time (msec)')
    axes.flat[0].set_ylabel(y_label)
    for ind, axis, name in zip(range(1, nrows * ncols), axes.flat[1:], list(dicts[0].keys())[:nrows * ncols - 1]):
        vals = [x[name] for x in dicts]
        if do_hists:
            axis.hist(vals)
        else:
            axis.bar(range(len(vals)), vals)
        axis.set_title(name)
        if ind % ncols == 0:
            axis.set_ylabel(y_label)
        if ind >= (nrows - 1) * ncols:
            axis.set_xlabel(x_label)

    fig.suptitle('Wall Times', y=1.03)
    fig.tight_layout()

    return fig


def calc_flops(timeline, analytical_ops, op_names):
    """
    Calculate FLOPS using duration per OP (from CUPTI) and analytical # of ops
    :param timeline:
    :param analytical_ops:
    :param op_names:
    :return: Total FLOPS (summed over all ops in op_names), dict['op_name'] = FLOPS
    """

    # TODO:

    # 1. Need dictionary with keys matching names in op_names that provides the total # of ops.

    # 2. parse timeline with load_hardware_trace_json(file)

    # 3. get all ops info with get_all_ops(trace_dic)

    # 4. get wall duration for each op in op_names with get_wall_duration()

    # 5. divide analytical ops by wall duration for each op in op_names.

    # 6. Return total FLOPS and dictionary with key=op_name and val= FLOPS.

    pass


def cal_nvprofs_flops(nvprof_csv, runtime=1):
    """
    Extract data from nvprof and calculate/return OPS, FLOPS.
    """
    p = re.compile(r'Device')
    with open(nvprof_csv) as f:
        skip_ln = 0
        while(True):
            line = f.readline()
            match = p.search(line)
            if match:
                fields = line
                skip_ln += 1
                break
            if skip_ln > 20:
                print('The provided file is missing nvprof headers!')
                break
            skip_ln += 1
    fields = fields.split(',')
    print('Extracted Fields: ')
    print(fields)
    arr = np.genfromtxt(nvprof_csv,skip_header=skip_ln, delimiter='Floating Point Operations(Single Precision)',
                    comments='==',dtype=None, encoding=None)
    print(arr[0,1].split(','))
    data = np.array([itm.split(',')[1:] for itm in arr[:,-1]]).astype(np.float64)
    Min, Max, Avg = data.T
# #     return Min, Max, Avg
#     return arr
    OPS = np.array([Min.sum(), Max.sum(), Avg.sum()])
    # Return OPS
    print('Min: %2.3e , Max: %2.3e, Avg: %2.3e  (Total Kernel OPS)'%(OPS[0],OPS[1],OPS[2]))
    # Return FLOPS
    FLOPS = OPS/runtime
    print('Min: %2.3e , Max: %2.3e, Avg: %2.3e  (FLOPS)'%(FLOPS[0], FLOPS[1], FLOPS[2]))
    return OPS, FLOPS


# http://imatge-upc.github.io/telecombcn-2016-dlcv/slides/D2L1-memory.pdf

def conv(inputs, params, bytesize):
    # NCHW not NHWC
    num_weights = np.prod(params['kernel'] + [params['features'], inputs[1]])
    outputs = tuple([inputs[0], params['features'], inputs[2] // params['stride'][0], inputs[3] // params['stride'][1]])
    mem = np.prod(outputs) * bytesize
    this_ops = np.prod(params['kernel'] + list(inputs[1:]) + [params['features']])
    return outputs, num_weights, mem, this_ops


def linear(inputs, params, bytesize):
    if len(inputs) == 4:
        inputs = (inputs[0], np.prod(inputs[1:]))
    outputs = (inputs[0], params['bias'])
    num_weights = params['bias'] + np.prod([params['bias'], inputs[1]]) + batch_norm(inputs)
    mem = np.prod(outputs) * bytesize
    this_ops = inputs[1] * params['weights']
    return outputs, num_weights, mem, this_ops


def pool(inputs, params, bytesize):
    outputs = (inputs[0], inputs[1], inputs[2] // params['stride'][0], inputs[3] // params['stride'][1])
    mem = np.prod(outputs) * bytesize
    return outputs, 0, mem, 0


def residual(orig_inputs, params, bytesize):
    weights = 0
    mem = 0
    ops = 0
    inputs = orig_inputs
    for layer_name, layer_params in list(params.items()):
        if not layer_name.startswith('conv'):
            continue
        outputs, curr_weights, curr_mem, curr_ops = conv(inputs, layer_params, bytesize)
        weights += curr_weights + batch_norm(outputs)
        mem += curr_mem
        ops += curr_ops
        # print('\t%s - %s, weights: %3.1e, memory: %3.1f MB, ops: %3.1e' % \
        # (layer_name, outputs, curr_weights, curr_mem / 1024**2, curr_ops))
        inputs = outputs
    # last conv for
    if outputs != orig_inputs:
        shortcut_parms = {"kernel": [1, 1], "features": outputs[1], "batch_norm": True, "stride": [1, 1]}
        orig_inputs, curr_weights, curr_mem, curr_ops = conv(orig_inputs, shortcut_parms, bytesize)
        weights += curr_weights
        mem += curr_mem
        ops += curr_ops
    return outputs, weights, mem, ops


def batch_norm(inputs):
    return 2 * inputs[1]


def calculate_network(inputs, network, is_fp16=False):
    bytesize = 4
    if is_fp16:
        bytesize = 2

    readouts = []
    tot_weights = 0
    tot_mem = np.prod(inputs) * bytesize
    tot_ops = 0

    print('Inputs: %s, memory: %3.1f MB' % (inputs, tot_mem / 1024 ** 2))
    for layer_name, layer_params in list(network.items()):
        # print('-------------------------')
        # print(layer_name, layer_params)
        if layer_params['type'] == 'convolutional':
            func = conv
        elif layer_params['type'] == 'pooling':
            func = pool
        elif layer_params['type'] in ['fully_connected', 'linear_output']:
            func = linear
        elif layer_params['type'] == 'residual':
            func = residual
        else:
            print('Unrecognized layer type ' + layer_params['type'])
            break
        outputs, weights, mem, this_ops = func(inputs, layer_params, bytesize)
        print('%s - %s, weights: %d, memory: %3.1f MB, ops: %3.1e' % (
        layer_name, outputs, weights, mem / 1024 ** 2, this_ops))
        inputs = outputs
        tot_ops += this_ops
        tot_mem += mem
        tot_weights += weights
        readouts.append([layer_name, outputs, weights, mem, this_ops])
    print('Total # of layers: %d,  weights: %3.1e, memory: %s MB, ops: %3.2e \n' % (len(network),
                                                                                    tot_weights,
                                                                                    tot_mem / 1024 ** 2,
                                                                                    tot_ops))
    return readouts, tot_weights, tot_mem, tot_ops
