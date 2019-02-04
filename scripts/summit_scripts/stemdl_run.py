"""
Created on 10/15/17.
@author: Numan Laanait, Mike Matheson
"""
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
#logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import numpy as np
import argparse
import json
import time
import sys
import os
import subprocess, shlex
import shutil
try:
   import horovod.tensorflow as hvd
except:
   print( "< ERROR > Could not import horovod module" )
   raise

from stemdl import runtime
from stemdl import io_utils

tf.logging.set_verbosity(tf.logging.ERROR)

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
    tf.set_random_seed( 1234 )
    np.random.seed( 4321 )

    # initiate horovod
    hvd.init()

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
    cmdline.add_argument( '--epochs_per_decay', default=None, type=float,
                         help="""Number of epochs per lr decay ( hyper parameter).""")
    cmdline.add_argument( '--scaling', default=None, type=float,
                         help="""Scaling (hyper parameter).""")
    cmdline.add_argument( '--bn_decay', default=None, type=float,
                         help="""Batch norm decay (hyper parameter).""")
    cmdline.add_argument('--save_epochs', default=0.5, type=float,
                         help="""Number of epochs to save checkpoint. """)
    cmdline.add_argument('--mode', default='train', type=str,
                         help="""train or eval (:validates from checkpoint)""")
    cmdline.add_argument('--cpu_threads', default=10, type=int,
                         help="""cpu threads per rank""")
    cmdline.add_argument('--accumulate_step', default=1, type=int,
                         help="""cpu threads per rank""")
    cmdline.add_argument( '--filetype', default=None, type=str,
                         help=""" lmdb or tfrecord""")
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
   
    
    FLAGS, unknown_args = cmdline.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            if hvd.rank( ) == 0 :
               print('<ERROR> Unknown command line arg: %s' % bad_arg)
        raise ValueError('Invalid command line arg(s)')

    # Load input flags
    if FLAGS.input_flags is not None :
       params = io_utils.get_dict_from_json( FLAGS.input_flags )
       params[ 'input_flags' ] = FLAGS.input_flags
    else :
       params = io_utils.get_dict_from_json('input_flags.json')
       params[ 'input_flags' ] = 'input_flags.json'

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
    if FLAGS.save_epochs is not None:
        params['epochs_per_saving'] = FLAGS.save_epochs
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
    params['nvme'] = FLAGS.nvme

    # Add other params
    params.setdefault( 'restart', False )

    checkpt_dir = params[ 'checkpt_dir' ]
    # Also need a directory within the checkpoint dir for event files coming from eval
    eval_dir = os.path.join( checkpt_dir, '_eval' )
    if hvd.rank() == 0:
        #print('ENVIRONMENT VARIABLES: %s' %format(os.environ))
        print( 'Creating checkpoint directory %s' % checkpt_dir )
    tf.gfile.MakeDirs( checkpt_dir )
    tf.gfile.MakeDirs( eval_dir )

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
    if FLAGS.scaling  is not None :
       hyper_params[ 'scaling' ] = FLAGS.scaling
    if FLAGS.epochs_per_decay is not None :
       hyper_params[ 'num_epochs_per_decay' ] = FLAGS.epochs_per_decay
    if FLAGS.bn_decay is not None :
       hyper_params[ 'batch_norm' ][ 'decay' ] = FLAGS.bn_decay
    
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

    # copy data to nvme
    #if FLAGS.nvme is not None:
   #     if hvd.rank() == 0:
   #         print('staging files on bb...')
   #     #params['data_dir']=os.environ.get('DATA_DIR')
   #     params = nvme_staging(params['data_dir'],params, mode=params['mode'])
    # train or evaluate
    if params['mode'] == 'train':
        runtime.train(network_config, hyper_params, params)
    elif params['mode'] == 'eval':
        params[ 'IMAGE_FP16' ] = False
        runtime.validate_ckpt(network_config, hyper_params, params, last_model=False, sleep=0)

def nvme_staging(data_dir, params, mode='train'):
    user = os.environ.get('USER')
    nvme_dir = '/mnt/bb/%s' %(user)
    if hvd.rank() == 0: print(os.listdir(nvme_dir))
    #cp_args = "cp -r %s/batch_%d.db %s/" %(params['data_dir'], hvd.rank(), nvme_dir)
    #if hvd.rank() == 0: print(cp_args)
    #cp_args = shlex.split(cp_args)
    #subprocess.run(cp_args, check=True)
    params['data_dir'] = nvme_dir
    return params        

if __name__ == '__main__':
    main()
