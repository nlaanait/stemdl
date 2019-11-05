"""
Created on 10/15/17.
@author: Numan Laanait, Mike Matheson
"""
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
import numpy as np
import argparse
import json
import time
import sys
import os
import copy
import subprocess, shlex
import shutil
import contextlib
import itertools
import random
from mpi4py import MPI

global world_rank
world_comm = MPI.COMM_WORLD
world_size = world_comm.Get_size()
world_rank = world_comm.Get_rank()

try:
   import horovod.tensorflow as hvd
except:
   print( "< ERROR > Could not import horovod module" )
   raise

from stemdl import runtime
from stemdl import io_utils

tf.logging.set_verbosity(tf.logging.ERROR)

def print_results(group_id, *args, **kwargs):
    print_f = open('search_%d.log' % group_id, 'a')
    with contextlib.redirect_stdout(print_f):
        print(*args, **kwargs)

def get_search_params(hyper_params, search_dic=dict()):
    lr = 10**(np.round(-np.linspace(2,5,num=4), decimals=2))
    initial = ['truncated_normal', 'glorot_uniform', 'variance_scaling', 
                'random_normal', 'random_uniform', 'xavier', 'he', 'uniform_unit_scaling']
    warm_up_steps = 10** np.round(np.linspace(1,4, num=5), decimals=1)
    warm_up_steps = warm_up_steps.astype(np.int)
    decay_steps = 10** np.round(np.linspace(3, 5, num=2), decimals=1)
    decay_steps = decay_steps.astype(np.int)
    weight_decay = 10 ** np.round(-np.linspace(3, 5, num=3), decimals=1)
    iterator = itertools.product(lr, initial, weight_decay, warm_up_steps, decay_steps)
    param_keys = ['initial_learning_rate' , 'initializer', 'weight_decay', 
                  'num_steps_in_warm_up', 'num_steps_per_decay']
    return iterator, param_keys

def get_group_search_params(iterator_list, group_id, num_comms=None):
    if num_comms is None:
        return
    chunk_size = len(iterator_list) // num_comms
    partitions = []
    for i in range(num_comms):
        part = slice(i * chunk_size, None) if i == num_comms-1 \
            else slice(i * chunk_size, (i+1) * chunk_size)
        partitions.append(part)
    group_partition = partitions[group_id]
    group_iterator = iterator_list[group_partition]
    random.shuffle(group_iterator) 
    return group_iterator 

def set_hyper_params(hyper_params, search_params, keys=[]):
    new_hyper_params = copy.deepcopy(hyper_params)
    for new_param, key in zip(search_params, keys):
        new_hyper_params[key] = new_param
        if key == 'initial_learning_rate':
            new_hyper_params['warm_up_max_learning_rate'] = new_hyper_params['initial_learning_rate'] * hyper_params['warm_up_max_learning_rate']\
                / hyper_params['initial_learning_rate'] 
    return hyper_params
        # try:
        #     hyper_params[key] = new_param
        # except KeyError:
        #     print('Key {} ')


def add_bool_argument(cmdline, shortname, longname=None, default=False, help=None):
    if longname is None:
        shortname, longname = None, shortname
    elif default == True:
        raise ValueError("""Boolean arguments that are True by default should not have short names.""")
    name = longname[2:]
    feature_parser = cmdline.add_mutually_exclusive_group(required=False)
    if shortname is not None:
        feature_parser.add_argument(shortname, '--'+name, dest=name, action='store_true', help=help, default=default)
    else:
        feature_parser.add_argument(           '--'+name, dest=name, action='store_true', help=help, default=default)
    feature_parser.add_argument('--no'+name, dest=name, action='store_false')
    return cmdline


def main():
    # tf.set_random_seed(1234)
    # np.random.seed(1234)

    cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Basic options
    cmdline.add_argument( '--batch_size', default=None, type=int,
                         help="""Size of each minibatch.""")
    cmdline.add_argument( '--log_frequency', default=None, type=int,
                         help="""Logging frequency.""")
    cmdline.add_argument( '--max_steps', default=None, type=int,
                         help="""Maximum steps.""")
    cmdline.add_argument( '--network_config', default=None, type=str,
                         help="""Neural net architecture.""")
    cmdline.add_argument( '--data_dir', default=None, type=str,
                         help="""Data directory [train/test].""")
    cmdline.add_argument( '--checkpt_dir', default=None, type=str,
                         help="""Checkpoint directory.""")
    cmdline.add_argument( '--input_flags', default=None, type=str,
                         help="""Input json.""")
    cmdline.add_argument( '--hyper_params', default=None, type=str,
                         help="""Hyper parameters.""")
    cmdline.add_argument( '--ilr', default=None, type=float,
                         help="""Initial learning rate ( hyper parameter).""")
    cmdline.add_argument( '--warm_steps', default=int(1e6), type=int,
                         help="""Number of Steps to do linear warm-up.""")
    cmdline.add_argument( '--save_steps', default=int(1e3), type=int,
                         help="""Number of Steps to save""")
    cmdline.add_argument( '--validate_steps', default=int(1e3), type=int,
                         help="""Number of Steps to validate.""")
    cmdline.add_argument( '--decay_steps', default=int(1e3), type=int,
                         help="""Number of steps per lr decay ( hyper parameter).""")
    cmdline.add_argument( '--summary_steps', default=int(1e3), type=int,
                         help="""Number of steps to save summaries.""")
    cmdline.add_argument( '--scaling', default=None, type=float,
                         help="""Scaling (hyper parameter).""")
    cmdline.add_argument( '--bn_decay', default=None, type=float,
                         help="""Batch norm decay (hyper parameter).""")
    cmdline.add_argument('--validate_epochs', default=1.0, type=float,
                         help="""Number of epochs to validate """)
    cmdline.add_argument('--mode', default='train', type=str,
                         help="""train or eval (:validates from checkpoint)""")
    cmdline.add_argument('--cpu_threads', default=10, type=int,
                         help="""cpu threads per rank""")
    cmdline.add_argument('--accumulate_step', default=0, type=int,
                         help="""cpu threads per rank""")
    cmdline.add_argument( '--filetype', default=None, type=str,
                         help=""" lmdb or tfrecord""")
    cmdline.add_argument( '--hvd_group', default=None, type=int,
                         help="""number of horovod message groups""")
    cmdline.add_argument( '--grad_ckpt', default=None, type=str,
                         help="""gradient-checkpointing:collection,memory,speed""")
    cmdline.add_argument( '--max_time', default=int(1e18), type=int,
                         help="""maximum time to run training loop""")
    cmdline.add_argument('--gpus_node', default=4, type=int,
                         help="""number of gpus per node""")
    cmdline.add_argument('--sub_comms', default=0, type=int,
                         help="""number of MPI subcommunicators to launch""")
    add_bool_argument( cmdline, '--fp16', default=None,
                         help="""Train with half-precision.""")
    add_bool_argument( cmdline, '--fp32', default=None,
                         help="""Train with single-precision.""")
    add_bool_argument( cmdline, '--restart', default=None,
                         help="""Restart training from checkpoint.""")
    add_bool_argument( cmdline, '--nvme', default=None,
                         help="""Copy data to burst buffer.""")
    add_bool_argument( cmdline, '--debug', default=None,
                         help="""Debug print commands.""")
    add_bool_argument( cmdline, '--hvd_fp16', default=None,
                         help="""horovod message compression""")
   
    
    FLAGS, unknown_args = cmdline.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            if hvd.rank( ) == 0 :
               print('<ERROR> Unknown command line arg: %s' % bad_arg)
        raise ValueError('Invalid command line arg(s)')

    # define MPI comms and initiate horovod
    if FLAGS.sub_comms != 0:
        group_id = world_comm.rank % FLAGS.sub_comms 
        sub_comm = MPI.COMM_WORLD.Split(color=group_id, key=world_comm.rank)
        group_rank = sub_comm.Get_rank()
        group_size = sub_comm.Get_size()
        hvd.init(comm=sub_comm)
        gpu_id = world_rank % FLAGS.gpus_node
    else:
        hvd.init()
        gpu_id = None

    # Load input flags
    if FLAGS.input_flags is not None :
       params = io_utils.get_dict_from_json( FLAGS.input_flags )
       params[ 'input_flags' ] = FLAGS.input_flags
    else :
       params = io_utils.get_dict_from_json('input_flags.json')
       params[ 'input_flags' ] = 'input_flags.json'
    params['no_jit'] = True 
    params[ 'start_time' ] = time.time( )
    params[ 'cmdline' ] = 'unknown'
    params['accumulate_step'] = FLAGS.accumulate_step
    if FLAGS.batch_size is not None :
        params[ 'batch_size' ] = FLAGS.batch_size
    if FLAGS.log_frequency is not None :
        params[ 'log_frequency' ] = FLAGS.log_frequency
    if FLAGS.max_steps is not None :
        params[ 'max_steps' ] = FLAGS.max_steps
    if FLAGS.network_config is not None :
        params[ 'network_config' ] = FLAGS.network_config
    if FLAGS.data_dir is not None :
        params[ 'data_dir' ] = FLAGS.data_dir
    if FLAGS.checkpt_dir is not None :
        params[ 'checkpt_dir' ] = FLAGS.checkpt_dir
    if FLAGS.hyper_params is not None :
        params[ 'hyper_params' ] = FLAGS.hyper_params
    if FLAGS.fp16 is not None :
        params[ 'IMAGE_FP16' ] = True
    if FLAGS.fp32 is not None :
        params[ 'IMAGE_FP16' ] = False
    if FLAGS.restart is not None :
        params[ 'restart' ] = True
    if FLAGS.validate_epochs is not None:
        params['epochs_per_validation'] = FLAGS.validate_epochs
    if FLAGS.mode == 'train':
        params['mode'] = 'train'
    if FLAGS.mode == 'eval':
        params['mode'] = 'eval'
    if FLAGS.cpu_threads is not None:
        params['IO_threads'] = FLAGS.cpu_threads
    if FLAGS.filetype is not None:
        params['filetype'] = FLAGS.filetype
    if FLAGS.debug is not None:
        params['debug'] = FLAGS.debug
    else: 
        params['debug'] = False
    params['save_step'] = FLAGS.save_steps 
    params['validate_step']= FLAGS.validate_steps 
    params['summary_step']= FLAGS.summary_steps 
    params['hvd_group'] = FLAGS.hvd_group
    params['max_time'] = FLAGS.max_time
    if FLAGS.hvd_fp16 is not None:
        params['hvd_fp16'] = hvd.Compression.fp16
    else: 
        params['hvd_fp16'] = hvd.Compression.none
    params['nvme'] = FLAGS.nvme
    params['grad_ckpt'] = FLAGS.grad_ckpt 

    # Add other params
    params.setdefault( 'restart', False )

    checkpt_dir = params[ 'checkpt_dir' ]
    eval_dir = os.path.join( checkpt_dir, '_eval' )
    if params[ 'gpu_trace' ] :
        if tf.gfile.Exists( params[ 'trace_dir' ] ) :
            print( 'Timeline directory %s exists' % params[ 'trace_dir' ] )
        else :
            print( 'Timeline directory %s created' % params[ 'trace_dir' ] )
            tf.gfile.MakeDirs( params[ 'trace_dir' ] )

    params['train_dir'] = checkpt_dir
    params['eval_dir'] = eval_dir
    # load network config file and hyper_parameters
    network_config = io_utils.load_json_network_config(params['network_config'])
    hyper_params = io_utils.load_json_hyper_params(params['hyper_params'])

    if FLAGS.ilr  is not None :
       hyper_params[ 'initial_learning_rate' ] = FLAGS.ilr
       #hyper_params[ 'initial_learning_rate' ] = 1e-5 
    if FLAGS.scaling  is not None :
       hyper_params[ 'scaling' ] = FLAGS.scaling
    if FLAGS.bn_decay is not None :
       hyper_params[ 'batch_norm' ][ 'decay' ] = FLAGS.bn_decay
    if FLAGS.warm_steps >= 1:
       hyper_params['warm_up'] = True
       hyper_params['num_steps_in_warm_up'] = FLAGS.warm_steps 
       hyper_params['num_steps_per_warm_up'] = FLAGS.warm_steps
    else: 
       hyper_params['warm_up'] = False 
       hyper_params['num_steps_in_warm_up'] = 1 
       hyper_params['num_steps_per_warm_up'] = 1 
    hyper_params['num_steps_per_decay'] = FLAGS.decay_steps 


    #cap max warm-up learning rate by ilr
    hyper_params["warm_up_max_learning_rate"] = hyper_params['initial_learning_rate'] * hvd.size()

    # print relevant params passed to training 
    if hvd.rank( ) == 0 :
       if os.path.isfile( 'cmd.log' ) :
          cmd = open( "cmd.log", "r" )
          cmdline = cmd.readline( )
          params[ 'cmdline' ] = cmdline

       print( "### hyper_params.json" )
       _input = json.dumps( hyper_params, indent=3, sort_keys=False)
       print( "%s" % _input )
       
       print("### params passed at CLI")
       _input = json.dumps(vars(FLAGS), indent=4)
       print("%s" % _input) 
    
    iterator, params_keys = get_search_params(hyper_params)
    iterator_list = list(iterator)
    # broadcast random shuffled search params per group
    if group_rank == 0:
        group_iter_list = get_group_search_params(iterator_list, group_id, num_comms=FLAGS.sub_comms)
    else:
        group_iter_list = None
    group_iter_list = sub_comm.bcast(group_iter_list)
    
    for search_param in group_iter_list:
        if params['debug']:
        # Ensure that ranks within group have the same params
            sub_comm.Barrier()
            print_results(group_id, 'group_id= %d, world rank=%d' % (group_id, world_rank), search_param)
            sub_comm.Barrier()
            world_comm.Barrier()
        hyper_params = set_hyper_params(hyper_params, search_param, keys=params_keys)
        if params['mode'] == 'train':
            # print_f = open('search_%d.log' % group_id, 'a')
            # with contextlib.redirect_stdout(print_f):
            val_results, train_results = runtime.train(network_config, hyper_params, params, gpu_id=gpu_id)
            if group_rank == 0:
                # print(group_id, 'Validation:\n', val_results, 'Training:\n', train_results)
                print_results(group_id, 'Validation:\n', val_results, 'Training:\n', train_results)
        
    # copy checkpoints from nvme
    if FLAGS.nvme is not None:
        if hvd.rank() == 0:
            print('copying files from bb...')
            nvme_staging(params['data_dir'],params)
    
def nvme_staging(data_dir, params):
    user = os.environ.get('USER')
    gpfs_ckpt_dir = os.environ.get('CKPT_DIR')
    #nvme_dir = '/mnt/bb/%s' %(user)
    #if hvd.rank() == 0: print(os.listdir(nvme_dir))
    cp_args = "cp -r %s %s" %(params['checkpt_dir'], gpfs_ckpt_dir)
    #if hvd.rank() == 0: print(cp_args)
    cp_args = shlex.split(cp_args)
    subprocess.run(cp_args, check=True)
    return         

if __name__ == '__main__':
    main()
