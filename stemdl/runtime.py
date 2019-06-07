"""
Created on 10/9/17.
@author: Numan Laanait, Michael Matheson
email: laanaitn@ornl.gov, mathesonm@ornl.gov
"""

import time
from datetime import datetime
import os
import sys
import re
import numpy as np
import math
from itertools import chain
from multiprocessing import cpu_count

#TF
import tensorflow as tf
from collections import OrderedDict
import horovod.tensorflow as hvd
from tensorflow.python.client import timeline
from tensorflow.contrib.compiler import xla

# stemdl
from . import network
from . import inputs
from . import optimizers
from . import lr_policies
from . import losses

tf.logging.set_verbosity(tf.logging.ERROR)

def tensorflow_version_tuple():
    v = tf.__version__
    major, minor, patch = v.split('.')
    return (int(major), int(minor), patch)


def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable


class TrainHelper(object):
    def __init__(self, params, saver, writer, net_ops, last_step=0, log_freq=1):
        self.params = params
        self.last_step = last_step
        self.net_ops = net_ops
        self.start_time = time.time()
        self.cumm_time = time.time()
        self.saver = saver
        self.writer = writer
        self.elapsed_epochs = self.last_step * self.params['batch_size'] * 1.0 * hvd.size() / \
                              self.params['NUM_EXAMPLES_PER_EPOCH']
        self.log_freq = log_freq

    def before_run(self):
        self.last_step +=1
        self.start_time = time.time()
        self.elapsed_epochs = self.last_step * self.params['batch_size'] * 1.0 * hvd.size() / \
                              self.params['NUM_EXAMPLES_PER_EPOCH']
        # call to hvd forces global namespace into class on purpose.

    def write_summaries(self, summary):
        if hvd.rank() == 0:
            with tf.summary.FileWriter(self.params['checkpt_dir']) as summary_writer:
                summary_writer.add_summary(summary, global_step=self.last_step)
        print_rank('Saved Summaries.')

    def save_checkpoint(self):
        pass

    def run_summary(self) :
        tfversion = tensorflow_version_tuple()
        print_rank( 'TensorFlow          ... %i.%i.%s' % tfversion )
        if 'LSB_JOBNAME' in os.environ :
           print_rank( 'job name            ... %s' % os.environ[ 'LSB_JOBNAME' ] )
        if 'LSB_JOBID' in os.environ :
           print_rank( 'job number          ... %s' % os.environ[ 'LSB_JOBID' ] )
        if 'LSB_OUTPUTFILE' in os.environ :
           print_rank( 'job output          ... %s' % os.environ[ 'LSB_OUTPUTFILE' ] )
        print_rank( 'number of ranks     ... %d' % hvd.size( ) )
        print_rank( 'network_config      ... %s' % self.params[ 'network_config' ] )
        print_rank( 'batch_size          ... %d' % self.params[ 'batch_size' ] )
        print_rank( '                    ... %d total' % ( self.params[ 'batch_size' ] * hvd.size( ) ) )
        print_rank( 'data type           ... %s' % ( 'fp16' if self.params[ 'IMAGE_FP16' ] else 'fp32' ) )
        print_rank( 'data_dir            ... %s' % self.params[ 'data_dir' ] )
        print_rank( 'input_flags         ... %s' % self.params[ 'input_flags' ] )
        print_rank( 'hyper_params        ... %s' % self.params[ 'hyper_params' ] )
        print_rank( 'checkpt_dir         ... %s' % self.params[ 'checkpt_dir' ] )
        print_rank( '' )
        print_rank( 'command line        ... %s' % self.params[ 'cmdline' ] )
        print_rank( '' )

    @staticmethod
    def save_trace(run_metadata, trace_dir, trace_step):
        # Writing trace to json file. open with chrome://tracing
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        with open( trace_dir + '/timeline_' + str( trace_step ) + '.ctf.' + str(hvd.rank()) + '.json', 'w') as f:
            f.write(trace.generate_chrome_trace_format( show_memory = True, show_dataflow = True ))
        print_rank('Run & Saved GPU Trace.')

    def log_stats(self, loss_value, learning_rate):
        self.nanloss(loss_value)
        if hvd.rank() == 0:
            t = time.time( )
            duration = t - self.start_time
            examples_per_sec = self.params['batch_size'] * hvd.size() / duration
            self.cumm_time = (time.time() - self.cumm_time)/self.log_freq
            flops = self.net_ops * examples_per_sec
            avg_flops = self.net_ops * self.params['batch_size'] * hvd.size() / self.cumm_time
            format_str = (
            'time= %.1f, step= %d, epoch= %2.2e, loss= %.2f, lr= %.2e, step_time= %2.2f sec, ranks= %d, examples/sec= %.1f, flops = %3.2e, average_time= %2.2f, average_flops= %3.3e')
            print_rank(format_str % ( t - self.params[ 'start_time' ],  self.last_step, self.elapsed_epochs,
                        loss_value, learning_rate, duration, hvd.size(), examples_per_sec, flops, self.cumm_time, avg_flops) )
            self.cumm_time = time.time()

    @staticmethod
    def nanloss(loss_value):
        if np.isnan(loss_value):
            print_rank('loss is nan... Exiting!')
            sys.exit(0)


def print_rank(*args, **kwargs):
    if hvd.rank() == 0:
        print(*args, **kwargs)


def train(network_config, hyper_params, params):
    """
    Train the network for a number of steps using horovod and asynchronous I/O staging ops.

    :param network_config: OrderedDict, network configuration
    :param hyper_params: OrderedDict, hyper_parameters
    :param params: dict
    :return: None
    """
    #########################
    # Start Session         #
    #########################
    # Config file for tf.Session()
    config = tf.ConfigProto(allow_soft_placement=params['allow_soft_placement'],
                           log_device_placement=params['log_device_placement'],
                           )
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.gpu_options.force_gpu_compatible = True
    config.intra_op_parallelism_threads = 6 
    config.inter_op_parallelism_threads = max(1, cpu_count()//6)
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
    # JIT causes gcc errors on dgx-dl and is built without on Summit.
    sess = tf.Session(config=config)


    ############################
    # Setting up Checkpointing #
    ###########################

    last_step = 0
    if params[ 'restart' ] :
       # Check if training is a restart from checkpoint
       ckpt = tf.train.get_checkpoint_state(params[ 'checkpt_dir' ] )
       if ckpt is None :
          print_rank( '<ERROR> Could not restart from checkpoint %s' % params[ 'checkpt_dir' ])
       else :
          last_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
          print_rank("Restoring from previous checkpoint @ step=%d" %last_step)

    global_step = tf.Variable(last_step, name='global_step',trainable=False)


    ############################################
    # Setup Graph, Input pipeline and optimizer#
    ############################################
    # Start building the graph

    # Setup data stream
    with tf.device(params['CPU_ID']):
        with tf.name_scope('Input') as _:
            if params['filetype'] == 'tfrecord':
                dset = inputs.DatasetTFRecords(params, dataset=params['dataset'], debug=False)
            elif params['filetype'] == 'lmdb':
                dset = inputs.DatasetLMDB(params, dataset=params['dataset'], debug=params['debug'])
            images, labels = dset.minibatch()
            # Staging images on host
            staging_op, (images, labels) = dset.stage([images, labels])

    with tf.device('/gpu:%d' % hvd.local_rank()):
        # Copy images from host to device
        gpucopy_op, (images, labels) = dset.stage([images, labels])
        IO_ops = [staging_op, gpucopy_op]

        ##################
        # Building Model#
        ##################

        # Build model, forward propagate, and calculate loss
        scope = 'model'
        summary = False
        if params['debug']:
            summary=True
        print_rank('Starting up queue of images+labels: %s,  %s ' % (format(images.get_shape()),
                                                                format(labels.get_shape())))

        with tf.variable_scope(scope,
                # Force all variables to be stored as float32
                custom_getter=float32_variable_storage_getter) as _:

            # Setup Neural Net
            if params['network_class'] == 'resnet':
                n_net = network.ResNet(scope, params, hyper_params, network_config, images, labels,
                                        operation='train', summary=summary, verbose=False)
            if params['network_class'] == 'cnn':
                n_net = network.ConvNet(scope, params, hyper_params, network_config, images, labels,
                                        operation='train', summary=summary, verbose=True)
            if params['network_class'] == 'fcdensenet':
                n_net = network.FCDenseNet(scope, params, hyper_params, network_config, images, labels,
                                        operation='train', summary=summary, verbose=True)
            if params['network_class'] == 'fcnet':
                n_net = network.FCNet(scope, params, hyper_params, network_config, images, labels,
                                        operation='train', summary=summary, verbose=True)
            
                
            ###### XLA compilation #########    
            #if params['network_class'] == 'fcdensenet':
            #    def wrap_n_net(*args):
            #        images, labels = args
            #        n_net = network.FCDenseNet(scope, params, hyper_params, network_config, images, labels,
            #                            operation='train', summary=False, verbose=True)
            #        n_net.build_model()
            #        return n_net.model_output
            #
            #    n_net.model_output = xla.compile(wrap_n_net, inputs=[images, labels])
            ##############################

            # Build it and propagate images through it.
            n_net.build_model()

            # calculate the total loss
            total_loss, loss_averages_op = losses.calc_loss(n_net, scope, hyper_params, params, labels, summary=summary)

            #get summaries, except for the one produced by string_input_producer
            if summary: summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)


        #######################################
        # Apply Gradients and setup train op #
        #######################################

        # get learning policy
        def learning_policy_func(step):
            return lr_policies.decay_warmup(params, hyper_params, step)
            ## TODO: implement other policies in lr_policies

        iter_size = params.get('accumulate_step', 0)
        skip_update_cond = tf.cast(tf.floormod(global_step, tf.constant(iter_size, dtype=tf.int32)), tf.bool)

        if params['IMAGE_FP16']:
            opt_type='mixed'
        else:
            opt_type=tf.float32
        # setup optimizer
        opt_dict = hyper_params['optimization']['params'] 
        train_opt, learning_rate = optimizers.optimize_loss(total_loss, hyper_params['optimization']['name'], 
                                opt_dict, learning_policy_func, run_params=params, hyper_params=hyper_params, iter_size=iter_size, dtype=opt_type, 
                                loss_scaling=hyper_params.get('loss_scaling',1.0), 
                                skip_update_cond=skip_update_cond,
                                on_horovod=True, model_scopes=n_net.scopes)  

    # Gather all training related ops into a single one.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    increment_op = tf.assign_add(global_step, 1)
    ema = tf.train.ExponentialMovingAverage(decay=0.75, num_updates=global_step)
    all_ops = tf.group(*([train_opt] + update_ops + IO_ops + [increment_op]))

    with tf.control_dependencies([all_ops]):
            train_op = ema.apply(tf.trainable_variables()) 
            # train_op = tf.no_op(name='train')

    ########################
    # Setting up Summaries #
    ########################

    # Stats and summaries
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    if hvd.rank() == 0:
        summary_writer = tf.summary.FileWriter(params['checkpt_dir'], sess.graph)
        # Add Summary histograms for trainable variables and their gradients
    if params['debug'] and hyper_params['network_type'] == 'inverter':
        predic = tf.transpose(n_net.model_output, perm=[0,2,3,1])
        output_summary = tf.summary.image("outputs", predic, max_outputs=4) 
        tf.summary.image("targets", tf.transpose(labels, perm=[0,2,3,1]), max_outputs=4)
        tf.summary.image("inputs", tf.transpose(tf.reduce_mean(images, axis=1, keepdims=True), perm=[0,2,3,1]), max_outputs=4)
    
    summary_merged = tf.summary.merge_all()

     ###############################
    # Setting up training session #
    ###############################

    #Initialize variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Sync
    print_rank('Syncing horovod ranks...')
    sync_op = hvd.broadcast_global_variables(0)
    sess.run(sync_op)
    
    # prefill pipeline first
    print_rank('Prefilling I/O pipeline...')
    for i in range(len(IO_ops)):
        sess.run(IO_ops[:i + 1])


    # Saver and Checkpoint restore
    checkpoint_file = os.path.join(params[ 'checkpt_dir' ], 'model.ckpt')
    saver = tf.train.Saver(max_to_keep=None, save_relative_paths=True)

    # Check if training is a restart from checkpoint
    if params['restart'] and ckpt is not None:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print_rank("Restoring from previous checkpoint @ step=%d" % last_step)

    # Train
    if hvd.rank() == 0:
        train_elf = TrainHelper(params, saver, summary_writer,  n_net.get_ops(), last_step=last_step, log_freq=params['log_frequency'])
    else:
        train_elf = TrainHelper(params, saver, None, n_net.get_ops(), last_step=last_step)

    if params['restart']:
        saveStep = train_elf.last_step + params['save_step']
        validateStep = train_elf.last_step + params['validate_step']
        summaryStep = train_elf.last_step + params['summary_step'] 
    else:
        saveStep =  params['save_step']
        validateStep = params['validate_step']
        summaryStep = params['summary_step']

    train_elf.run_summary( )
    maxSteps  = params[ 'max_steps' ]
    logFreq   = params[ 'log_frequency' ]
    traceStep = params[ 'trace_step' ]

    while train_elf.last_step < maxSteps :
        train_elf.before_run()

        doLog   = train_elf.last_step % logFreq  == 0
        doSave  = train_elf.last_step >= saveStep 
        doSumm  = train_elf.last_step > summaryStep 
        doTrace = train_elf.last_step == traceStep and params['gpu_trace']
        if train_elf.last_step == 1 and params['debug']:
            if hvd.rank() == 0:
                summary = sess.run([train_op,  summary_merged])[-1]
                train_elf.write_summaries( summary )
        if not doLog and not doSave and not doTrace and not doSumm:
            sess.run(train_op)
        elif doLog and not doSave :
            _, loss_value, lr = sess.run( [ train_op, total_loss, learning_rate ] )
            train_elf.log_stats( loss_value, lr )
        elif doLog and doSave :
            _, summary, loss_value, lr = sess.run( [ train_op, summary_merged, total_loss, learning_rate ])
            train_elf.log_stats( loss_value, lr )
            train_elf.write_summaries( summary )
            if hvd.rank( ) == 0 :
                saver.save(sess, checkpoint_file, global_step=train_elf.last_step)
                print_rank('Saved Checkpoint.')
            saveStep += params['save_step']
        elif doSumm and params['debug']:
            if hvd.rank() == 0:
                summary = sess.run([train_op,  summary_merged])[-1]
                train_elf.write_summaries( summary )
            summaryStep += params['summary_step'] 
        elif doSave :
            #summary = sess.run([train_op,  summary_merged])[-1]
            #train_elf.write_summaries( summary )
            if hvd.rank( ) == 0 :
                saver.save(sess, checkpoint_file, global_step=train_elf.last_step)
                print_rank('Saved Checkpoint.')
            saveStep += params['save_step']
        elif doTrace :
            sess.run(train_op, options=run_options, run_metadata=run_metadata)
            train_elf.save_trace(run_metadata, params[ 'trace_dir' ], params[ 'trace_step' ] )
            train_elf.before_run()
        # Here we do validation:
        #if train_elf.elapsed_epochs > next_validation_epoch:
        if train_elf.last_step > validateStep:
            validate(network_config, hyper_params, params, sess, dset)
            validateStep += params['validate_step']
            #next_validation_epoch += params['epochs_per_validation']


def validate(network_config, hyper_params, params, sess, dset, num_batches=10):
    """
    Runs validation with current weights
    :param params:
    :param hyper_params:
    :param network_config:
    :param sess:
    :param num_batches: default 100.
    :return:
    """
    print_rank("Running Validation ..." )
    with tf.device(params['CPU_ID']):
        # Get Test data
        dset.set_mode(mode='eval')
        images, labels = dset.minibatch()
        # Staging images on host
        staging_op, (images, labels) = dset.stage([images, labels])

    with tf.device('/gpu:%d' % hvd.local_rank()):
        # Copy images from host to device
        gpucopy_op, (images, labels) = dset.stage([images, labels])
        IO_ops = [staging_op, gpucopy_op]

    scope = 'model'
    summary = False

    # prefill pipeline first
    print_rank('Prefilling I/O pipeline...')
    for i in range(len(IO_ops)):
        sess.run(IO_ops[:i + 1])

    with tf.variable_scope(scope, reuse=True) as _:
        # Setup Neural Net
        params['IMAGE_FP16'] = False
        if images.dtype != tf.float32:
            images = tf.cast(images, tf.float32)
        # Setup Neural Net
        if params['network_class'] == 'resnet':
            n_net = network.ResNet(scope, params, hyper_params, network_config, images, labels,
                                      operation='eval', summary=False, verbose=False)
        if params['network_class'] == 'cnn':
            n_net = network.ConvNet(scope, params, hyper_params, network_config, images, labels,
                                     operation='eval', summary=False, verbose=False)
        if params['network_class'] == 'fcdensenet':
            n_net = network.FCDenseNet(scope, params, hyper_params, network_config, images, labels,
                                     operation='eval', summary=False, verbose=False)
        if params['network_class'] == 'fcnet':
            n_net = network.FCNet(scope, params, hyper_params, network_config, images, labels,
                                    operation='eval', summary=summary, verbose=True)

        # Build it and propagate images through it.
        n_net.build_model()

    # Calculate predictions
    if hyper_params['network_type'] == 'regressor' or hyper_params['network_type'] == 'classifier':
        labels_shape = labels.get_shape().as_list()
        layer_params={'bias':labels_shape[-1], 'weights':labels_shape[-1],'regularize':False}
        logits = fully_connected(n_net, layer_params, params['batch_size'],
                                name='linear',reuse=None)
    else:
        pass
        #TODO: implement prediction layer for hybrid network

    # Do evaluation
    if hyper_params['network_type'] == 'regressor':
        validation_error = tf.losses.mean_squared_error(labels, predictions=logits, reduction=tf.losses.Reduction.NONE)
        # Average validation error over the batches
        errors = np.array([sess.run(validation_error) for _ in range(num_batches)])
        errors = errors.reshape(-1, params['NUM_CLASSES'])
        avg_errors = errors.mean(0)
        print_rank('Validation MSE: %s' % format(avg_errors))
    elif hyper_params['network_type'] == 'classifier':
        labels = tf.argmax(labels, axis=1)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        in_top_1_op = tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32)
        in_top_5_op = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
        eval_ops = [in_top_1_op, in_top_5_op, cross_entropy]
        output = np.array([sess.run(eval_ops) for _ in range(num_batches)])
        accuracy = output[:,:2]
        val_loss = output[:,-1]
        accuracy = accuracy.sum(axis=(0,-1))/(num_batches*params['batch_size'])*100
        val_loss = val_loss.sum()/(num_batches*params['batch_size'])
        print_rank('Validation Accuracy (.pct), Top-1: %2.2f , Top-5: %2.2f, Loss: %2.2f' %(accuracy[0], accuracy[1], val_loss))
    elif hyper_params['network_type'] == 'hybrid':
        #TODO: implement evaluation call for hybrid network
        print('not implemented')
    elif hyper_params['network_type'] == 'inverter':
        loss_params = hyper_params['loss_function']
        if loss_params['type'] == 'MSE_PAIR':
            errors = tf.losses.mean_pairwise_squared_error(tf.cast(labels, tf.float32), tf.cast(n_net.model_output, tf.float32))
            loss_label= loss_params['type'] 
        else: 
            loss_label= 'ABS_DIFF'
            errors = tf.losses.absolute_difference(tf.cast(labels, tf.float32), tf.cast(n_net.model_output, tf.float32), reduction=tf.losses.Reduction.MEAN)
        errors = tf.expand_dims(errors,axis=0)
        error_averaging = hvd.allreduce(errors)
        errors = np.array([sess.run([IO_ops,errors])[-1] for i in range(dset.num_samples // params['batch_size'])])
        # errors = np.array([sess.run([IO_ops,errors])[-1] for i in range(dset.num_samples)])
        # errors = tf.reduce_mean(errors)
        # avg_errors = hvd.allreduce(tf.expand_dims(errors, axis=0))
        # error = sess.run(avg_errors)
        # print_rank('Validation Reconstruction Error %s: %3.3e' % (loss_label, errors.mean()))
        print('rank=%d, Validation Reconstruction Error %s: %3.3e' % (hvd.rank(), loss_label, errors.mean()))
        tf.summary.scalar("Validation_loss_label_%s" %hvd.rank() , tf.constant(errors.mean()))


def validate_ckpt(network_config, hyper_params, params, num_batches=None,
                    last_model= False, sleep=-1):
    """
    Runs evaluation with current weights
    :param params:
    :param hyper_params:
    :param network_config:
    :param num_batches: default 100.
    :params sleep: number of seconds to sleep. for single eval pass sleep<0.
    :return:
    """
    #########################
    # Start Session         #
    #########################
    # Config file for tf.Session()
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False,
                            )
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.intra_op_parallelism_threads = 1
    # config.inter_op_parallelism_threads = 12
    sess = tf.Session(config=config)

    # Get Test data

    with tf.device(params['CPU_ID']):
        with tf.name_scope('Input') as _:
            if params['filetype'] == 'tfrecord':
                dset = inputs.DatasetTFRecords(params, dataset=params['dataset'], debug=False)
            elif params['filetype'] == 'lmdb':
                dset = inputs.DatasetLMDB(params, dataset=params['dataset'], debug=params['debug'])
            images, labels = dset.minibatch()
            # Staging images on host
            staging_op, (images, labels) = dset.stage([images, labels])

    with tf.device('/gpu:%d' % hvd.local_rank()):
        # Copy images from host to device
        gpucopy_op, (images, labels) = dset.stage([images, labels])
        IO_ops = [staging_op, gpucopy_op]

        scope='model'
        with tf.variable_scope(
                scope,
                # Force all variables to be stored as float32
                custom_getter=float32_variable_storage_getter) as _:
            # Setup Neural Net
            if params['network_class'] == 'resnet':
                n_net = network.ResNet(scope, params, hyper_params, network_config, tf.cast(images, tf.float32), labels,
                                        operation='eval_ckpt', summary=False, verbose=False)
            if params['network_class'] == 'cnn':
                n_net = network.ConvNet(scope, params, hyper_params, network_config, tf.cast(images, tf.float32), labels,
                                        operation='eval_ckpt', summary=False, verbose=False)
            if params['network_class'] == 'fcdensenet':
                n_net = network.FCDenseNet(scope, params, hyper_params, network_config, tf.cast(images, tf.float32),
                                            labels, operation='eval_ckpt', summary=False, verbose=True)
            if params['network_class'] == 'fcnet':
                n_net = network.FCNet(scope, params, hyper_params, network_config, images, labels,
                                        operation='eval_ckpt', summary=False, verbose=True)
            # Build it and propagate images through it.
            n_net.build_model()

        # Calculate predictions
        #if hyper_params['network_type'] == 'regressor' or hyper_params['network_type'] == 'classifier':
        #    labels_shape = labels.get_shape().as_list()
        #    layer_params={'bias':labels_shape[-1], 'weights':labels_shape[-1],'regularize':False}
        #    logits = fully_connected(n_net, layer_params, params['batch_size'],
        #                            name='linear',reuse=None)
        #else:
        #    pass

        # Initialize variables
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # Sync
        sync_op = hvd.broadcast_global_variables(0)
        sess.run(sync_op)

        # prefill pipeline first
        print_rank('Prefilling I/O pipeline...')
        for i in range(len(IO_ops)):
            sess.run(IO_ops[:i + 1])

        # restore from moving averages
        ema = tf.train.ExponentialMovingAverage(0.9999)
        vars_to_restore = ema.variables_to_restore()
        # saver = tf.train.Saver(var_list=vars_to_restore)
        saver = tf.train.Saver()

        # Find models in checkpoint directory
        dirs = np.array(os.listdir(params['checkpt_dir']))
        pattern = re.compile("meta")
        steps = np.array([bool(re.search(pattern,itm)) for itm in dirs])
        saved_steps = dirs[steps]
        model_steps = np.array([int(itm.split('.')[1].split('-')[-1]) for itm in saved_steps])
        model_steps = np.sort(model_steps)
        ckpt_paths = [os.path.join(params['checkpt_dir'], "model.ckpt-%s" % step) for step in model_steps]

        if last_model:
            ckpt_paths = [ckpt_paths[-1]]
            model_steps = [model_steps[-1]]

        # Validate Models
        for ckpt, last_step in zip(ckpt_paths, model_steps):
            #
            saver.restore(sess, os.path.join(params['checkpt_dir'],"model.ckpt-%s" %format(last_step)))
            print_rank("Restoring from previous checkpoint @ step=%d" % last_step)

            # Validate model
            # TODO: add hybrid validation and check that it works correctly for previous
            if hyper_params['network_type'] == 'regressor':
                validation_error = tf.losses.mean_squared_error(labels, predictions=logits, reduction=tf.losses.Reduction.NONE)
                # Average validation error over batches
                errors = np.array([sess.run([IO_ops, validation_error])[-1] for _ in range(num_batches)])
                errors = errors.reshape(-1, params['NUM_CLASSES'])
                avg_errors = errors.mean(0)
                print_rank('Validation MSE: %s' % format(avg_errors))
            elif hyper_params['network_type'] == 'classifier':
                # Average validation accuracies over batches
                label = tf.argmax(labels, axis=1)
                in_top_1_op = tf.cast(tf.nn.in_top_k(logits, label, 1), tf.float32)
                in_top_5_op = tf.cast(tf.nn.in_top_k(logits, label, 5), tf.float32)
                eval_ops = [in_top_1_op,in_top_5_op]
                output = np.array([sess.run([IO_ops,eval_ops])[-1] for _ in range(num_batches)])
                accuracy = output.sum(axis=(0,-1))/(num_batches*params['batch_size'])*100
                print_rank('Validation Accuracy (.pct), Top-1: %2.2f , Top-5: %2.2f' %(accuracy[0], accuracy[1]))
            elif hyper_params['network_type'] == 'hybrid':
                pass
            elif hyper_params['network_type'] == 'inverter':
                loss_params = hyper_params['loss_function']
                if params['output']:
                    output = tf.cast(n_net.model_output, tf.float32)
                    print('output shape',output.get_shape().as_list()) 
                    output_dir = os.path.join(os.getcwd(),'outputs')
                    if num_batches is not None:
                        num_samples = num_batches
                    else:
                        num_samples = dset.num_samples
                    for idx in range(num_samples):
                        output_arr, label_arr = sess.run([IO_ops, n_net.model_output, labels])[-2:]
                        #label_arr = sess.run([IO_ops, labels])[-1]
                        np.save(os.path.join(output_dir,'label_%d_%d_%s.npy' % (idx, hvd.rank(), format(last_step))), label_arr)
                        np.save(os.path.join(output_dir,'output_%d_%d_%s.npy' % (idx, hvd.rank(), format(last_step))), output_arr)
                else:
                    if loss_params['type'] == 'MSE_PAIR':
                        errors = tf.losses.mean_pairwise_squared_error(tf.cast(labels, tf.float32), tf.cast(n_net.model_output, tf.float32))
                        loss_label= loss_params['type'] 
                    else: 
                        loss_label= 'ABS_DIFF'
                        errors = tf.losses.absolute_difference(tf.cast(labels, tf.float32), tf.cast(n_net.model_output, tf.float32), reduction=tf.losses.Reduction.MEAN)
                    errors = tf.expand_dims(errors,axis=0)
                    error_averaging = hvd.allreduce(errors)
                    if num_batches is not None:
                        num_samples = num_batches
                    else:
                        num_samples = dset.num_samples
                    error = np.array([sess.run([IO_ops,error_averaging])[-1] for i in range(num_samples)])
                    print_rank('Validation Reconstruction Error %s: %3.3e' % (loss_label, error.mean()))
            if sleep < 0:
                break
            else:
                print_rank('sleeping for %d s ...' % sleep)
                time.sleep(sleep)
