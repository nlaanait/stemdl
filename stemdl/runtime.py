"""
Created on 10/9/17.
@author: Numan Laanait.
email: laanaitn@ornl.gov
"""

import time
from datetime import datetime
import tensorflow as tf
import re
import numpy as np
from . import network
from . import inputs
import os
from collections import OrderedDict


class _LoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime stats."""
    def __init__(self, flags, total_loss, num_gpus, last_step=0):
        self.flags = flags
        self.total_loss = total_loss
        self.num_gpus = num_gpus
        self.last_step = last_step

    def begin(self):
        self._step = -1 + self.last_step
        self._start_time = time.time()
        self.epoch = 0.

    def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(self.total_loss)  # Asks for loss value.

    def after_run(self, run_context, run_values):
        if self._step % self.flags.log_frequency == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time

            loss_value = run_values.results
            examples_per_sec = self.flags.log_frequency * self.flags.batch_size * self.num_gpus / duration
            sec_per_batch = float(duration / self.flags.log_frequency)
            elapsed_epochs = self.num_gpus * self._step * self.flags.batch_size * 1.0 / self.flags.NUM_EXAMPLES_PER_EPOCH
            self.epoch += elapsed_epochs
            format_str = ('%s: step = %d, epoch = %2.2e, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch/gpu)')
            print(format_str % (datetime.now(), self._step, elapsed_epochs, loss_value,
                                examples_per_sec, sec_per_batch))


def _add_loss_summaries(total_loss, losses, flags, summaries=False):
    """
    Add summaries for losses in model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    :param flags:
    :param total_loss:
    :param losses:
    :return: loss_averages_op
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss;
    if summaries:
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            loss_name = re.sub('%s_[0-9]*/' % flags.worker_name, '', l.op.name)
            tf.summary.scalar(loss_name + ' (raw)', l)
            tf.summary.scalar(loss_name, loss_averages.average(l))

    return loss_averages_op

# TODO: double-for loop in python. Ouch! Needs to go.
# Just compute running average in net building block. L#200-244.
def _average_gradients(worker_grads):
    """Calculate the average gradient for each shared variable across all workers.
    This function essentially synchronizes all workers.
    Args:
    worker_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each worker.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all workers.
    """
    if len(worker_grads) == 1:
        return worker_grads[0]
    grads_list = []

    for i in range(len(worker_grads[0])):
        dummy=[]
        for grad in worker_grads:
            dummy.append(grad[i][0])
        grads_list.append(tf.stack(dummy))

    # Average over the 'worker' dimension.
    grad_tensor = [tf.reduce_mean(grad, axis=0) for grad in grads_list]

    # Getting shared variables
    variables = [itm[1] for itm in worker_grads[0]]
    average_grads = [(grad, var) for grad, var in zip(grad_tensor, variables)]
    return average_grads


def get_optimizer(flags, hyper_params, global_step):
    """
    Setups an optimizer object and returns it.
    :param flags: tf.app.flags
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
    num_batches_per_epoch = flags.NUM_EXAMPLES_PER_EPOCH / flags.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    ramp_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_RAMP)
    ramp_up_steps = int(num_batches_per_epoch * NUM_EPOCHS_IN_WARM_UP)

    # Decay the learning rate exponentially based on the number of steps.
    def ramp():
        # lr = INITIAL_LEARNING_RATE*LEARNING_RATE_DECAY_FACTOR**(-global_step/ramp_steps)
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, ramp_steps, LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        lr = INITIAL_LEARNING_RATE ** 2 * tf.pow(lr, tf.constant(-1.))
        return tf.cast(lr, tf.float32)

    def decay():
        lr = tf.train.exponential_decay(WARM_UP_LEARNING_RATE_MAX, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        return lr

    if hyper_params['warm_up']:
        LEARNING_RATE = tf.cond(global_step < ramp_up_steps, ramp, decay)
        LEARNING_RATE = tf.cond(LEARNING_RATE <= WARM_UP_LEARNING_RATE_MAX, lambda: LEARNING_RATE, decay)
    else:
        LEARNING_RATE = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR, staircase=True)

    # Summarize learning rate
    tf.summary.scalar('learning_rate', LEARNING_RATE)

    # optimizer
    if hyper_params['optimization'] == 'SGD':
        opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        return opt
    if hyper_params['optimization'] == 'Momentum':
        opt = tf.train.MomentumOptimizer(LEARNING_RATE, momentum= hyper_params['momentum'])
        return opt

    # Default is ADAM
    opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1= hyper_params['momentum'])
    return opt


# TODO: Implement another train function that synchronizes only when the average loss rate changes "considerably".
def train(network_config, hyper_params, data_path, flags, num_GPUS=1):
    """
    Train the network for a number of steps.
    # At each step (global_step):
    # 1. propagate examples through the neural net.
    # 2. calculate the losses/gradients for each worker.
    # 3. average over gradients.
    # 4. update the neural net weights.
    # 5. repeat.
    :param network_config: OrderedDict, network configuration
    :param hyper_params: OrderedDict, hyper_parameters
    :param flags: tf.app.flags
    :param num_GPUS: int, default 1.
    :param data_path: string, path to data.
    :return: None
    """
    # Check if training is a restart from checkpoint
    ckpt = tf.train.get_checkpoint_state(flags.train_dir)
    if ckpt is None:
        last_step = 0
    else:
        last_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])

    # Only neural net ops will live on GPU.
    # Everything else (variable initialization, placement, updates) is on the host.

    ##################################
    # Building Model and replicating #
    ##################################

    # Start building the graph
    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Setup data stream
        with tf.name_scope('Input') as _:
            filename_queue = tf.train.string_input_producer([data_path], num_epochs=flags.num_epochs)
            # pass the filename_queue to the inputs classes to decode
            dset = inputs.DatasetTFRecords(filename_queue, flags)
            image, label = dset.decode_image_label()

        # setup optimizer
        opt = get_optimizer(flags, hyper_params, global_step)

        # Build model, forward propagate, and calculate loss for each worker.
        worker_grads = []
        worker_ops = []
        worker_total_loss = []
        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            for gpu_id in range(num_GPUS):
                # Flag to only generate summaries on the first device.
                summary = gpu_id == 0
                with tf.device('/gpu:%d' % gpu_id):
                    with tf.name_scope('%s_%d' % (flags.worker_name, gpu_id)) as scope:

                        # Process images and generate examples batch
                        images, labels = dset.train_images_labels_batch(image, label, distort=flags.train_distort,
                                                                        noise_min=0.0, noise_max=0.25,
                                                                        random_glimpses='normal', geometric=True)

                        print('Starting up queue of images+labels: %s,  %s ' % (format(images.get_shape()),
                                                                                format(labels.get_shape())))

                        # Setup Neural Net
                        n_net = network.ConvNet(scope, flags, global_step, hyper_params, network_config, images, labels,
                                             operation='train', summary=summary)

                        # Build it and propagate images through it.
                        n_net.build_model()

                        # calculate the total loss
                        n_net.get_loss()

                        # Assemble all of the losses.
                        losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
                        regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

                        # Calculate the total loss for the current worker
                        total_loss = tf.add_n(losses+regularization, name='total_loss')

                        # Accumulate total across all workers
                        worker_total_loss.append(total_loss)

                        # Generate summaries for the losses and get corresponding op
                        loss_averages_op = _add_loss_summaries(total_loss, losses, flags, summaries=summary)

                        # get summaries, except for the one produced by string_input_producer
                        # TODO: figure out the summaries nonsense.
                        if summary: summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the current data batch
                        with tf.control_dependencies([loss_averages_op]):
                            grads_vars = opt.compute_gradients(total_loss)

                        # TODO: Do running average here.
                        # Accumulate gradients across all workers.
                        worker_grads.append(grads_vars)

                        # Accumulate extra non-standard operations across workers
                        worker_ops.append(n_net.get_misc_ops())

    #######################################
    # Synchronizing across model replicas #
    #######################################

        # average over gradients.
        avg_gradients = _average_gradients(worker_grads)

        # Apply gradients to trainable variables
        apply_gradient_op = opt.apply_gradients(avg_gradients, global_step=global_step)

        # Add Summary histograms for trainable variables and their gradients
        # TODO: using the default summary save in MonitoredTrainingSession causes all kinds of "useless"
        # summaries to be saved. Disable it and instantiate a summary saver object for additional control. See eval().

        for grad, var in avg_gradients:
            summaries.append(tf.summary.histogram(var.op.name, var))
            summaries.append(tf.summary.histogram(var.op.name+'/gradients', grad))
        summary_merged = tf.summary.merge_all()

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(hyper_params['moving_average_decay'], global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

        # Gather all training related ops into a single one.
        with tf.control_dependencies([apply_gradient_op, variable_averages_op, tf.group(*worker_ops)]):
            train_op = tf.no_op(name='train')

        ###############################
        # Setting up training session #
        ###############################

        # Config file for tf.Session()
        config = tf.ConfigProto(allow_soft_placement=flags.allow_soft_placement,
                                log_device_placement=flags.log_device_placement)

        #calculate average loss and setup logger
        avg_total_loss = tf.reduce_mean(worker_total_loss)
        logHook = _LoggerHook(flags, avg_total_loss, num_GPUS, last_step=last_step)

        # Stats and summaries
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary_writer = tf.summary.FileWriter(flags.train_dir)

        # Start Training Session
        with tf.train.MonitoredTrainingSession(checkpoint_dir=flags.train_dir,
                                               hooks=[tf.train.StopAtStepHook(last_step=flags.max_steps),
                                                      tf.train.NanTensorHook(avg_total_loss),logHook], config=config,
                                               save_summaries_steps=None, save_summaries_secs=None,
                                               save_checkpoint_secs=300) as mon_sess:
            while not mon_sess.should_stop():
                if logHook._step % flags.save_frequency == 0:
                    # Train, Record stats and save summaries
                    _, sum_merged = mon_sess.run([train_op, summary_merged], options= run_options,
                                                 run_metadata=run_metadata)
                    summary_writer.add_run_metadata(run_metadata, 'step %s' % format(logHook._step),
                                                    global_step=logHook._step)
                    summary_writer.add_summary(sum_merged, global_step=logHook._step)
                    print('Running Stats and Saving Summaries...')
                else:
                    # Just train
                    mon_sess.run(train_op)

            summary_writer.close()


def eval(network_config, hyper_params, data_path, flags, num_GPUS=1):
    """
        Evaluate the network for a number of steps.
        # 1. load the neural net from the checkpoint directory.
        # 2. Evaluate neural net predictions.
        # 3. repeat.
        :param network_config: OrderedDict, network configuration
        :param hyper_params: OrderedDict, hyper_parameters
        :param flags: tf.app.flags
        :param num_GPUS: int, default 1.
        :param data_path: string, path to data.
        :return: None
    """
    if num_GPUS == 0:
        device = '/cpu:0'
        cpu_bound = True
    else:
        device = '/gpu:%d' % (num_GPUS - 1)
        gpu_id = num_GPUS - 1
        cpu_bound = False

    with tf.device(device):
        with tf.Graph().as_default() as g:
            # Setup data stream
            with tf.name_scope('Input_Eval') as _:
                filename_queue = tf.train.string_input_producer([data_path])
                # pass the filename_queue to the inputs classes to decode
                dset = inputs.DatasetTFRecords(filename_queue, flags)
                image, label = dset.decode_image_label()
                # distort images and generate examples batch
                images, labels = dset.eval_images_labels_batch(image, label, noise_min=0.05, noise_max=0.25,
                                                               distort=flags.eval_distort, random_glimpses='normal',
                                                               geometric=False)

            with tf.variable_scope(tf.get_variable_scope(), reuse=None):

                # Build the model and forward propagate
                # Force the evaluation of MSE if doing regression
                if hyper_params['network_type'] == 'regressor':
                    hyper_params['loss_function']['type'] = 'MSE'

                # Setup Neural Net
                n_net = network.ConvNet('worker_0/', flags, 0, hyper_params, network_config, images, labels,
                                        operation='eval', summary=False)

                # Build it and propagate images through it.
                n_net.build_model()

                # get the output and the error
                prediction = n_net.model_output

            # Initialize a dictionary of evaluation ops
            eval_ops = OrderedDict()
            eval_ops['prediction'] = prediction
            if hyper_params['network_type'] == 'regressor':
                MSE_op = tf.losses.mean_squared_error(labels, predictions=prediction, reduction=tf.losses.Reduction.NONE)
                eval_ops['errors'] = [MSE_op]
                eval_ops['errors_labels'] = 'Mean-Squared Error'
            if hyper_params['network_type'] == 'classifier':
                # Calculate top-1 and top-5 error
                labels = tf.argmax(labels, axis=1)
                in_top_1_op = tf.cast(tf.nn.in_top_k(prediction, labels, 1), tf.float32)
                in_top_5_op = tf.cast(tf.nn.in_top_k(prediction, labels, 5), tf.float32)
                eval_ops['errors'] = [in_top_1_op, in_top_5_op]
                eval_ops['errors_labels'] = ['Top-1 Precision', 'Top-5 Precision']

            # Initiate restore object
            saver = tf.train.Saver()

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(flags.eval_dir, g)

            while True:
                if hyper_params['network_type'] == 'classifier':
                    eval_classify(flags, saver, summary_writer, eval_ops, summary_op, labels, cpu_bound=cpu_bound,
                                  gpu_id=gpu_id, save_to_disk=True)
                else:
                    eval_regress(flags, saver, summary_writer, eval_ops, summary_op,labels, cpu_bound=cpu_bound,
                                  gpu_id=gpu_id, save_to_disk=True)
                if flags.run_once:
                    break
                time.sleep(flags.eval_interval_secs)


def eval_regress(flags, saver, summary_writer, eval_ops, summary_op, labels, cpu_bound=True, gpu_id=0, save_to_disk=True):
    """
    Helper function to eval() for regression tasks: preprocess predictions, save summaries, and start queue runners.
    :param flags: tf.app.flags
    :param saver: tf.train.Saver object, restores neural net model.
    :param summary_writer: a tf.summary.FileWriter, writes tensorboard summaries to disk.
    :param eval_ops: tf.op, evaluation ops.
    :param summary_op: tf.op, summary operations.
    :param cpu_bound: bool, whether to run evaluation solely on the host, default True.
    :param gpu_id: int, device id where to run evaluation, default 0.
    :param save_to_disk: bool, whether to save to disk.
    :return: None
    """
    # Config file for tf.Session()
    config = tf.ConfigProto(allow_soft_placement=flags.allow_soft_placement,
                            log_device_placement=flags.log_device_placement)

    if cpu_bound:
        device = '/cpu:0'
        config.device_count= {'GPU': 0}
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.001
    else:
        device = '/gpu:%d' % gpu_id

    with tf.device(device):
        with tf.Session(config=config) as sess:
            # Restore Model from checkpoint
            ckpt = tf.train.get_checkpoint_state(flags.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                # train/model.ckpt-0,
                # Extract global_step
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No valid Checkpoint File was found!!')
                return

            # TODO: Extend tf.MonitoredTrainSession for evaluations session to minimize all this garbage collection.

            # Start the queue runners for the input stream.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                     start=True))

                num_evals = int(np.ceil(flags.num_examples / flags.batch_size))
                step = 0

                # Allocate arrays
                sorted_errors = np.array([])
                predictions = np.array([])
                angles_arr = np.array([])

                # Begin evaluation
                start_time = time.time()
                while step < num_evals and not coord.should_stop():
                    # evaluate predictions
                    angle, predic, err = sess.run([labels, eval_ops['prediction'], eval_ops['errors']])
                    sorted_errors = np.append(sorted_errors, err)
                    angles_arr = np.append(angles_arr, angle)
                    predictions = np.append(predictions, predic)
                    step += 1

                # Reformatting the output arrays
                sorted_errors = np.reshape(sorted_errors,(-1, flags.OUTPUT_DIM))
                angles_arr = np.reshape(angles_arr, (-1, flags.OUTPUT_DIM))
                predictions = np.reshape(predictions, (-1, flags.OUTPUT_DIM))

                # saved_vars = []
                # for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                #     if 'moving_' in var.name:
                #         saved_vars.append(var)
                #
                # print([var.name for var in saved_vars])
                # output = sess.run(saved_vars)
                # print(output)

                # Get mean error
                mean_errors = np.mean(sorted_errors,axis=0)

                # Get time it took to evaluate
                print('Took %.3f seconds to evaluate %d images' % (time.time() - start_time, flags.num_examples))

                # Save predictions and labels to disk
                if save_to_disk:
                    fname = '%s_%s.npy' % ('predictions', format(datetime.now()).split(' ')[-1])
                    np.save(os.path.join(flags.eval_dir, fname), predictions, allow_pickle=False)
                    fname = '%s_%s.npy' % ('angles', format(datetime.now()).split(' ')[-1])
                    np.save(os.path.join(flags.eval_dir, fname), angles_arr, allow_pickle=False)

                # Print Model Outputs and Summarize in Tensorboard
                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                output_labels = flags.output_labels.split(';')
                for mean_error, output_label in zip(mean_errors, output_labels):
                    print('%s: %s = %.3f' % (datetime.now(), format(output_label), mean_error))
                    summary.value.add(tag=output_label, simple_value=mean_error)

                summary_writer.add_summary(summary, global_step)

            except Exception as e:
                coord.request_stop(e)

            # Kill the input queue threads
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=1)


def eval_classify(flags, saver, summary_writer, eval_ops, summary_op, labels, cpu_bound=True,
                 gpu_id=0, save_to_disk=True):
    """
    Helper function to eval() for classification tasks: preprocess predictions, save summaries, and start queue runners.
    :param flags: tf.app.flags
    :param saver: tf.train.Saver object, restores neural net model.
    :param summary_writer: a tf.summary.FileWriter, writes tensorboard summaries to disk.
    :param eval_ops: tf.op, evaluation ops.
    :param summary_op: tf.op, summary operations.
    :param labels: tf.tensor, labels
    :param cpu_bound: bool, whether to run evaluation solely on the host, default True.
    :param gpu_id: int, device id where to run evaluation, default 0.
    :param save_to_disk: bool, whether to save to disk.
    :return: None
    """
    # Config file for tf.Session()
    config = tf.ConfigProto(allow_soft_placement=flags.allow_soft_placement,
                            log_device_placement=flags.log_device_placement)

    if cpu_bound:
        device = '/cpu:0'
        config.device_count = {'GPU': 0}
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.001
    else:
        device = '/gpu:%d' % gpu_id

    with tf.device(device):
        with tf.Session(config=config) as sess:
            # Restore Model from checkpoint
            ckpt = tf.train.get_checkpoint_state(flags.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                # train/model.ckpt-0,
                # Extract global_step
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No valid Checkpoint File was found!!')
                return

            # TODO: Extend tf.MonitoredTrainSession for evaluations session to minimize all this garbage collection.

            # Start the queue runners for the input stream.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                     start=True))

                num_evals = int(np.ceil(flags.num_examples / flags.batch_size))
                step = 0

                # In the case of multiple outputs, we sort the predictions per output.
                # Allocate arrays
                sorted_errors = [np.zeros(shape=(flags.NUM_CLASSES,)) for _ in enumerate(eval_ops['errors'])]
                true_counts = [0 for _ in enumerate(eval_ops['errors'])]


                # Begin evaluation
                start_time = time.time()
                while step < num_evals and not coord.should_stop():
                    # evaluate predictions
                    errors_outputs = sess.run(eval_ops['errors'])
                    # figure out classes present in batch
                    classes = np.array(sess.run([labels])).flatten()
                    uniq_cls, uniq_indx, uniq_cts = np.unique(classes, return_index=True, return_counts=True)
                    # collect errors
                    for i, _ in enumerate( eval_ops['errors']):
                        # Get total of correct predictions
                        true_counts[i] += np.sum(errors_outputs[i])
                        # Sort predictions amongst classes
                        zeroes = np.zeros_like(sorted_errors[i])
                        zeroes[uniq_cls] = errors_outputs[i][uniq_indx]
                        sorted_errors[i] += zeroes
                    step += 1

                # Total sample count
                total_sample_count = step*flags.batch_size

                # Averaging over the output arrays
                for i, _ in enumerate(eval_ops['errors']):
                    true_counts[i] /= float(total_sample_count)
                    sorted_errors[i] /= float(step)

                # Get time it took to evaluate
                print('Took %.3f seconds to evaluate %d images' % (time.time() - start_time, flags.num_examples))

                # TODO: Save precisions to disk
                if save_to_disk:
                    pass
                    # fname = '%s_%s.npy' % ('top_1_precision_per_class', format(datetime.now()).split(' ')[-1])
                    # np.save(os.path.join(flags.eval_dir, fname), sorted_errors[0], allow_pickle=False)
                    #
                    # fname = '%s_%s.npy' % ('top_5_precision_per_class', format(datetime.now()).split(' ')[-1])
                    # np.save(os.path.join(flags.eval_dir, fname), sorted_errors[1], allow_pickle=False)

                # Print Model Outputs and Summarize in Tensorboard
                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                output_labels = flags.output_labels.split(';')
                for true_count, output_label in zip(true_counts, output_labels):
                    print('%s: %s = %.3f' % (datetime.now(), format(output_label), true_count))
                    summary.value.add(tag=output_label, simple_value=true_count)

                summary_writer.add_summary(summary, global_step)

            except Exception as e:
                coord.request_stop(e)

            # Kill the input queue threads
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=1)


def set_flags(checkpt_dir, eval_dir, batch_size=64, data_dir=None):
    """
    Sets flags that could change from one run to the next
    :param checkpt_dir: checkpoint directory.
    :param batch_size: as it says.
    :param eval_dir: evaluation directory.
    :param data_dir: as it says.
    :return:
    """
    tf.app.flags.DEFINE_string('train_dir', checkpt_dir, """Directory where to write event logs and checkpoint.""")
    tf.app.flags.DEFINE_string('eval_dir', eval_dir, """Directory where to write event logs during evaluation.""")
    tf.app.flags.DEFINE_integer('batch_size', batch_size, """Number of images to process in a batch.""")
    tf.app.flags.DEFINE_string('data_dir', data_dir,"""Directory where data tfrecords is located""")