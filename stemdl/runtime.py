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

# stemdl
from . import network
from . import inputs


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
    def __init__(self, params, saver, writer, net_ops, last_step=0):
        self.params = params
        self.last_step = last_step
        self.net_ops = net_ops
        self.start_time = time.time()
        self.saver = saver
        self.writer = writer

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
            flops = self.net_ops * examples_per_sec
            format_str = (
            'time= %.1f, step= %d, epoch= %2.2e, loss= %.2f, lr= %.2e, step_time= %2.2f sec, ranks= %d, examples/sec= %.1f, flops = %3.2e')
            print_rank(format_str % ( t - self.params[ 'start_time' ],  self.last_step, self.elapsed_epochs,
                        loss_value, learning_rate, duration, hvd.size(), examples_per_sec, flops) )

    @staticmethod
    def nanloss(loss_value):
        if np.isnan(loss_value):
            print_rank('loss is nan... Exiting!')
            sys.exit(0)


def _add_loss_summaries(total_loss, losses, summaries=False):
    """
    Add summaries for losses in model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    :param params:
    :param total_loss:
    :param losses:
    :return: loss_averages_op
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    # loss_averages_op = loss_averages.apply([total_loss])

    # Attach a scalar summary to all individual losses and the total loss;
    if summaries:
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            # loss_name = re.sub('%s_[0-9]*/' % params['worker_name'], '', l.op.name)
            tf.summary.scalar(l.op.name + ' (raw)', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def get_optimizer(params, hyper_params, global_step):
    """
    Setups an optimizer object and returns it.
    :param params: dict
    :param hyper_params: dict, with hyper-parameters
    :return: optimizer
    """
    NUM_EPOCHS_PER_DECAY = hyper_params['num_epochs_per_decay']
    NUM_EPOCHS_PER_RAMP = hyper_params['num_epochs_per_ramp']
    NUM_EPOCHS_IN_WARM_UP = hyper_params['num_epochs_in_warm_up']
    INITIAL_LEARNING_RATE = hyper_params['initial_learning_rate']
    LEARNING_RATE_DECAY_FACTOR = hyper_params['learning_rate_decay_factor']
    WARM_UP_LEARNING_RATE_MAX = hyper_params['warm_up_max_learning_rate']

    # Set parameters that affect the learning rate.
    num_batches_per_epoch = params['NUM_EXAMPLES_PER_EPOCH'] / params['batch_size'] / hvd.size( )
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    ramp_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_RAMP)
    ramp_up_steps = int(num_batches_per_epoch * NUM_EPOCHS_IN_WARM_UP)

    # Decay/ramp the learning rate exponentially based on the number of steps.
    def ramp():
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, ramp_steps, LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        lr = INITIAL_LEARNING_RATE ** 2 * tf.pow(lr, tf.constant(-1.))
        lr = tf.minimum(lr,WARM_UP_LEARNING_RATE_MAX)
        return lr

    def decay(lr_current):
        lr = tf.train.exponential_decay(lr_current, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        return lr

    if hyper_params['warm_up']:
        LEARNING_RATE = tf.cond(global_step < ramp_up_steps, ramp, lambda: decay(ramp()))
    else:
        LEARNING_RATE = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR, staircase=True)

    #   learning rate schedule (2500)
    # TODO: Pass schedule through input_flags.json
    lr_steps  = [          5,    20,  100,  250,  1100,  2400,  5000,  7500 ]    # steps
    lr_values = [ 0.01, 0.02,  0.04, 0.10, 0.15, 0.075, 0.020,  0.01,  0.001 ]

    # Summarize learning rate
    tf.summary.scalar('learning_rate', LEARNING_RATE)

    # optimizer
    if hyper_params['optimization'] == 'SGD':
        opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)

    if hyper_params['optimization'] == 'Momentum':
        opt = tf.train.MomentumOptimizer(LEARNING_RATE, momentum= hyper_params['momentum'], use_nesterov=True)

    else:
        # Default is ADAM
        opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1= hyper_params['momentum'])

    opt = hvd.DistributedOptimizer(opt)
    return opt, LEARNING_RATE


def loss_func(model, hyper_params, var_scope, summary=False):
    # Build the forward model
    n_net = model
    n_net.build_model()

    #TODO: it's cleaner to just leave out output layer from network.py and do it Here
    # For multi-task stuff...

    # logits = n_net.model_output
    # if logits.dtype != tf.float32:
    #     logits = tf.cast(logits, tf.float32)

    n_net.get_loss()
    losses = tf.get_collection(tf.GraphKeys.LOSSES)
    regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    # loss = tf.losses.mean_squared_error(labels, predictions=logits,
    #                              reduction=tf.losses.Reduction.MEAN)

    # Calculate the total loss for the current worker
    total_loss = tf.add_n(losses + regularization, name='total_loss')

    #Generate summaries for the losses and get corresponding op
    loss_averages_op = _add_loss_summaries(total_loss, losses, summaries=summary)

    return total_loss


def print_rank(*args, **kwargs):
    if hvd.rank() == 0:
        print(*args, **kwargs)


def larc_gradients(grads_vars, hyper_params, scale):
    """
    From Sean and Houston (NVIDIA)
    Rescales gradient of each weight based on L2-norm value of the latter
    :param grads_vars:
    :param hyper_params:
    :param scale: loss scaling factor
    :return: rescaled grads_vars
    """
    for idx, (g, v) in enumerate(grads_vars):
        if g is not None:
            g /= scale
            v_norm = tf.norm(tensor=v, ord=2)
            g_norm = tf.norm(tensor=g, ord=2)

            larc_local_lr = tf.cond(
                pred=tf.logical_and(tf.not_equal(v_norm, tf.constant(0.0)),
                                    tf.not_equal(g_norm, tf.constant(0.0))),
                true_fn=lambda: hyper_params['LARC_eta'] * v_norm / g_norm,
                false_fn=lambda: hyper_params['LARC_epsilon'])

            if hyper_params['LARC_mode'] == "scale":
                effective_lr = larc_local_lr
            else:
                effective_lr = tf.minimum(larc_local_lr, 1.0)

            # rescale gradients
            grads_vars[idx] = (tf.scalar_mul(effective_lr, g), v)

    return grads_vars


def get_grads_vars_layer_indices(grads_vars, scopes):
    ind_dict = OrderedDict()
    for scope in scopes:
        p = re.compile(scope.name)
        ind_list = []
        for (ind, grad) in enumerate(grads_vars):
            if p.search(grad[1].name):
                ind_list.append(ind)
        ind_dict[scope.name] = ind_list
    return ind_dict


def lsa_gradients(grads_vars, scale):
    """
    Adaptive gradient updates
    :param grads_vars:
    :param hyper_params:
    :return:
    """

    g_norm_list = [tf.norm(grad[0]/scale) for grad in grads_vars]
    new_grads_vars = [((1 + tf.log1p(g_norm**(-1)))*grad[0]/scale, grad[1])
                      for g_norm, grad in zip(g_norm_list, grads_vars)]
    return new_grads_vars


def lsal_gradients(grads_vars, scopes, scale):
    """
    Adaptive layer-specific gradient updates
    :param scopes: scopes of each layer
    :return:
    """
    new_grads_vars = []
    layer_indices = get_grads_vars_layer_indices(grads_vars, scopes)
    for layer in layer_indices.keys():
        ind_list = layer_indices[layer]
        if len(ind_list) >= 1:
            layer_grads = [grads_vars[ind][0] for ind in ind_list]
            layer_vars = [grads_vars[ind][1] for ind in ind_list]
            grad_vec = tf.concat([tf.expand_dims(tf.reshape(grad, [-1]), 0) for grad in layer_grads], 1)
            grad_norm_inv = tf.norm(grad_vec/scale)**(-1)
            new_grads_vars_layer = [((1 + tf.log1p(grad_norm_inv))*grad/scale, var)
                                    for grad, var in zip(layer_grads, layer_vars)]
            new_grads_vars.append(new_grads_vars_layer)

    return list(chain.from_iterable(new_grads_vars))


def train_horovod_mod(network_config, hyper_params, params):
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
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = int(cpu_count()/6)
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
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
            dset = inputs.DatasetTFRecords(params, dataset=params['dataset'])
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
        scope = 'horovod'
        summary = False

        print_rank('Starting up queue of images+labels: %s,  %s ' % (format(images.get_shape()),
                                                                format(labels.get_shape())))

        with tf.variable_scope(
                'horovod',
                # Force all variables to be stored as float32
                custom_getter=float32_variable_storage_getter) as _:

            # Setup Neural Net
            if params['network_class'] == 'resnet':
                n_net = network.ResNet(scope, params, hyper_params, network_config, images, labels,
                                        operation='train', summary=False, verbose=False)
            if params['network_class'] == 'cnn':
                n_net = network.ConvNet(scope, params, hyper_params, network_config, images, labels,
                                        operation='train', summary=False, verbose=False)
            if params['network_class'] == 'fcdensenet':
                n_net = network.FCDenseNet(scope, params, hyper_params, network_config, images, labels,
                                        operation='train', summary=False, verbose=True)

            # Build it and propagate images through it.
            n_net.build_model()

            # calculate the total loss
            n_net.get_loss()


            #Assemble all of the losses.
            losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
            regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)


            # Calculate the total loss for the current worker
            total_loss = tf.add_n(losses + regularization, name='total_loss')

            #Generate summaries for the losses and get corresponding op
            loss_averages_op = _add_loss_summaries(total_loss, losses, summaries=summary)

            #get summaries, except for the one produced by string_input_producer
            if summary: summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)


        #######################################
        # Apply Gradients and setup train op #
        #######################################

        # setup optimizer
        opt, learning_rate = get_optimizer(params, hyper_params, global_step)

         # Apply gradients to trainable variables
        if params['IMAGE_FP16']:
            # scale the losses
            scaling = hyper_params['scaling']
            # Calculate the gradients for the current data batch
            with tf.control_dependencies([loss_averages_op]):
                # compute gradients
                grads_vars = opt.compute_gradients(tf.cast(total_loss * scaling, tf.float32))
                # update gradients
                if hyper_params['LSAL']:
                    new_grads_vars = lsal_gradients(grads_vars, n_net.scopes, scaling)
                # change update rules
                elif hyper_params['LARC']:
                    new_grads_vars = larc_gradients(grads_vars, hyper_params, scaling)
                else:
                    new_grads_vars = [(grads[0]/scaling,grads[1]) for grads in grads_vars]

            # apply gradients
            apply_gradient_op = opt.apply_gradients(new_grads_vars, global_step=global_step)
        else:
            with tf.control_dependencies([loss_averages_op]):
                apply_gradient_op = opt.minimize(total_loss, gate_gradients=tf.train.Optimizer.GATE_NONE)


        # Gather all training related ops into a single one.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        all_ops = tf.group(*([apply_gradient_op]+update_ops+IO_ops))

        with tf.control_dependencies([all_ops]):
                train_op = tf.no_op(name='train')

    #########################
    # Setting up Summaries #
    #########################

    # Stats and summaries
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    if hvd.rank() == 0:
        summary_writer = tf.summary.FileWriter(params['checkpt_dir'], sess.graph)
        # Add Summary histograms for trainable variables and their gradients
    summary_merged = tf.summary.merge_all()


     ###############################
    # Setting up training session #
    ###############################

    #Initialize variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Sync
    sync_op = hvd.broadcast_global_variables(0)
    sess.run(sync_op)

    # prefill pipeline first
    print_rank('Prefilling I/O pipeline...')
    for i in range(len(IO_ops)):
        sess.run(IO_ops[:i + 1])


    # Saver and Checkpoint restore
    checkpoint_file = os.path.join(params[ 'checkpt_dir' ], 'model.ckpt')
    saver = tf.train.Saver(max_to_keep=None)

    # Check if training is a restart from checkpoint
    if params['restart'] and ckpt is not None:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print_rank("Restoring from previous checkpoint @ step=%d" % last_step)

    # Train
    if hvd.rank() == 0:
        train_elf = TrainHelper(params, saver, summary_writer,  n_net.get_ops(), last_step=last_step)
    else:
        train_elf = TrainHelper(params, saver, None, n_net.ops, last_step=last_step)

    next_validation_epoch = params['epochs_per_validation']
    next_checkpoint_epoch = params['epochs_per_saving']

    train_elf.run_summary( )
    maxSteps  = params[ 'max_steps' ]
    logFreq   = params[ 'log_frequency' ]
    traceStep = params[ 'trace_step' ]

    while train_elf.last_step < maxSteps :
        train_elf.before_run()

        doLog   = train_elf.last_step % logFreq  == 0
        doSave  = train_elf.elapsed_epochs > next_checkpoint_epoch
        doTrace = train_elf.last_step == traceStep and params['gpu_trace']

        if not doLog and not doSave and not doTrace :
           sess.run(train_op)
        elif doLog and not doSave :
           loss_value, lr = sess.run( [ train_op, total_loss, learning_rate ] )[ -2: ]
           train_elf.log_stats( loss_value, lr )
        elif doLog and doSave :
           summary, loss_value, lr = sess.run( [ train_op, summary_merged, total_loss, learning_rate ] )[ -3: ]
           train_elf.log_stats( loss_value, lr )
           train_elf.write_summaries( summary )
           if hvd.rank( ) == 0 :
              saver.save(sess, checkpoint_file, global_step=train_elf.last_step)
              print_rank('Saved Checkpoint.')
           next_checkpoint_epoch += params['epochs_per_saving']
        elif doSave :
           summary = sess.run( [ train_op, summary_merged ] )[ -1 ]
           train_elf.write_summaries( summary )
           if hvd.rank( ) == 0 :
              saver.save(sess, checkpoint_file, global_step=train_elf.last_step)
              print_rank('Saved Checkpoint.')
           next_checkpoint_epoch += params['epochs_per_saving']
        elif doTrace :
           sess.run(train_op, options=run_options, run_metadata=run_metadata)
           train_elf.save_trace(run_metadata, params[ 'trace_dir' ], params[ 'trace_step' ] )

        # Here we do validation:
        if train_elf.elapsed_epochs > next_validation_epoch:
            # do validation over 300 batches.
            validate(hyper_params, params, network_config, sess, dset)
            next_validation_epoch += params['epochs_per_validation']


def validate(network_config, hyper_params, params, sess, dset, num_batches=300):
    """
    Runs validation with current weights
    :param params:
    :param hyper_params:
    :param network_config:
    :param sess:
    :param num_batches: default 100.
    :return:
    """
    print_rank("Running Validation over %d batches..." % num_batches)
    # Get Test data
    dset.set_mode(mode='test')
    images, labels = dset.minibatch()

    with tf.variable_scope('horovod', reuse=True) as _:
        # Setup Neural Net
        n_net = network.ResNet('horovod', params, hyper_params, network_config, tf.cast(images, tf.float32),
                               labels, operation='eval', summary=False)

        # Build it and propagate images through it.
        n_net.build_model()

        logits = n_net.get_output()

        if hyper_params['network_type'] == 'regressor':
            validation_error = tf.losses.mean_squared_error(labels, predictions=logits, reduction=tf.losses.Reduction.NONE)

            # Average validation error over the batches
            errors = np.array([sess.run(validation_error) for _ in range(num_batches)])
            errors = errors.reshape(-1, params['NUM_CLASSES'])
            avg_errors = errors.mean(0)
            print_rank('Validation MSE: %s' % format(avg_errors))
        else:
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


def validate_ckpt(network_config, hyper_params, params,num_batches=300,
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
            dset = inputs.DatasetTFRecords(params, dataset=params['dataset'], mode='test')
            images, labels = dset.minibatch()
            # Staging images on host
            staging_op, (images, labels) = dset.stage([images, labels])

    with tf.device('/gpu:%d' % hvd.local_rank()):
        # Copy images from host to device
        gpucopy_op, (images, labels) = dset.stage([images, labels])
        IO_ops = [staging_op, gpucopy_op]

    with tf.variable_scope('horovod', reuse=tf.AUTO_REUSE) as scope:
        # Setup Neural Net
        n_net = network.ResNet(scope, params, hyper_params, network_config, tf.cast(images, tf.float32),
                               labels, operation='eval_ckpt', summary=False)

        # Build it and propagate images through it.
        n_net.build_model()

        logits = n_net.get_output()

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
        if hyper_params['network_type'] == 'regressor':
            validation_error = tf.losses.mean_squared_error(labels, predictions=logits, reduction=tf.losses.Reduction.NONE)
            # Average validation error over batches
            errors = np.array([sess.run(validation_error) for _ in range(num_batches)])
            errors = errors.reshape(-1, params['NUM_CLASSES'])
            avg_errors = errors.mean(0)
            print_rank('Validation MSE: %s' % format(avg_errors))
        else:
            # Average validation accuracies over batches
            label = tf.argmax(labels, axis=1)
            in_top_1_op = tf.cast(tf.nn.in_top_k(logits, label, 1), tf.float32)
            in_top_5_op = tf.cast(tf.nn.in_top_k(logits, label, 5), tf.float32)
            eval_ops = [in_top_1_op,in_top_5_op]
            output = np.array([sess.run([IO_ops,eval_ops])[-1] for _ in range(num_batches)])
            accuracy = output.sum(axis=(0,-1))/(num_batches*params['batch_size'])*100
            print_rank('Validation Accuracy (.pct), Top-1: %2.2f , Top-5: %2.2f' %(accuracy[0], accuracy[1]))
        if sleep < 0:
            break
        else:
            print_rank('sleeping for %d s ...' % sleep)
            time.sleep(sleep)
