"""
Created on 03/27/18.
@author: Numan Laanait.
email: laanaitn@ornl.gov
"""
from collections import OrderedDict
import json
import sys

def load_hardware_trace_json(file):
    """
    Reads a timeline.json file output by Tensorflow/libcupti and returns and OrderedDict object
    :param file: .json file.
    :return: OrderedDict
    """
    with open(file, mode='r') as f:
        def _as_ordered_dict(val):
            return OrderedDict(val)

        def _as_list(val):
            return list(val)

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
        traceEvents = trace_dic['traceEvents']
    except KeyError:
        print('Not valid GPU trace dict object.')
        sys.exit()
    all_ops = []
    for trace in traceEvents:
        try:
            if trace['cat'] == 'Op':
                all_ops.append(trace)
        except KeyError:
            pass
    return all_ops


def get_unique_ops_names(all_ops):
    '''
    Find unique op names.
    Params:
    all_ops: list, of dictionary of all operations.
    Return: list of unique op names.
    '''
    return set(op['name'] for op in all_ops)


def get_wall_duration(op_names, all_ops, pid_list=[11, 7, 13, 15, 9]):
    '''
    Calculates wall duration for each op in op_names.
    Params:
    op_names: list (str), names of ops of interest.
    pid_list: list (str), names of pid to include.
    all_ops: output of get_all_ops().
    Return:
    total wall duration (summed over all ops in op_names), dict['op_name'] = wall duration.
    '''
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

    print('Wall Duration (ms): %4.3f' % total_dur)
    return total_dur, op_dict

def calc_flops(timeline, analytical_ops, op_names):
    '''
    Calculate FLOPS using duration per OP (from CUPTI) and analytical # of ops
    :param timeline:
    :param analytical_ops:
    :param op_names:
    :return: Total FLOPS (summed over all ops in op_names), dict['op_name'] = FLOPS
    '''

    #TODO:

    #1. Need dictionary with keys matching names in op_names that provides the total # of ops.

    #2. parse timeline with load_hardware_trace_json(file)

    #3. get all ops info with get_all_ops(trace_dic)

    #4. get wall duration for each op in op_names with get_wall_duration(op_names, all_ops, pid_list=[11, 7, 13, 15, 9])

    #5. divide analytical ops by wall duration for each op in op_names.

    #6. Return total FLOPS and dictionary with key=op_name and val= FLOPS.

    pass

