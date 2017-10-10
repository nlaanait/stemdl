"""
Created on 10/9/17.
@author: Numan Laanait.
email: laanaitn@ornl.gov
"""

import time
from datetime import datetime
import tensorflow as tf
import re
from . import network
from . import inputs


class _LoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime stats."""
    def __init__(self, flags, total_loss, num_gpus):
        self.flags = flags
        self.total_loss = total_loss
        self.num_gpus = num_gpus

    def begin(self):
        self._step = -1
        self._start_time = time.time()

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
            elapsed_epochs = self.num_gpus * self._step * self.flags.batch_size / self.flags.NUM_EXAMPLES_PER_EPOCH
            format_str = ('%s: step = %d, epoch = %2.2e, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (datetime.now(), self._step, elapsed_epochs, loss_value,
                                examples_per_sec, sec_per_batch))


def _add_loss_summaries(total_loss, losses, flags):
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

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        loss_name = re.sub('%s_[0-9]*/' % flags.worker_name, '', l.op.name)
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


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
    average_grads = []
    for grad_and_vars in zip(*worker_grads):
        # Note that each grad_and_vars looks like the following:
        #((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the worker.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'worker' dimension- to be average over
            grads.append(expanded_g)

    # Average over the 'worker' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # All the variables are shared so just return references from the first worker.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
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
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        ramp_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        lr = INITIAL_LEARNING_RATE ** 2 * tf.pow(lr, tf.constant(-1.))
        return tf.cast(lr, tf.float32)

    def decay():
        lr = tf.train.exponential_decay(WARM_UP_LEARNING_RATE_MAX,
                                        global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        return lr

    LEARNING_RATE = tf.cond(global_step < ramp_up_steps, ramp, decay)
    LEARNING_RATE = tf.cond(LEARNING_RATE <= WARM_UP_LEARNING_RATE_MAX,
                            lambda: LEARNING_RATE,
                            decay)

    # Summarize learning rate
    tf.summary.scalar('learning_rate', LEARNING_RATE)

    # optimizer
    if hyper_params['optimization'] == 'SGD':
        opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        return opt
    # Default is ADAM
    opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    return opt


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

    # Only neural net ops will live on GPU.
    # Everything else (variable initialization, placement, updates) is on the host.
    # Start building the graph

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Setup data stream
        with tf.variable_scope('Input') as scope:
            # Add queue runner to the graph
            filename_queue = tf.train.string_input_producer([data_path], num_epochs=flags.num_epochs)

            # pass the filename_queue to the inputs classes to decode
            dset = inputs.DatasetTFRecords(filename_queue, flags)
            image, label = dset.decode_image_label()

            # Process images and generate examples batch
            images, labels = dset.train_images_labels_batch(image, label, distort=True, noise_min=0.02,
                                                            noise_max=0.15,
                                                            random_glimpses='normal', geometric=True)

            print('Starting up queue of images+labels: %s,  %s ' % (format(images.get_shape()),
                                                                format(labels.get_shape())))

        # setup optimizer
        opt = get_optimizer(flags, hyper_params, global_step)

        # Build model, forward propagate, and calculate loss for each worker.
        worker_grads = []
        worker_ops = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_GPUS):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (flags.worker_name, i)) as scope:

                        # Setup Neural Net
                        n_net = network.ConvNet(flags, global_step, hyper_params, network_config, images, labels,
                                             operation='train')
                        # Build it
                        n_net.build_model()

                        # calculate the loss
                        # losses, total_loss = n_net.get_loss(scope)
                        n_net.get_loss()

                        # Assemble all of the losses.
                        losses = tf.get_collection('losses', scope)

                        # Calculate the total loss for the current worker
                        total_loss = tf.add_n(losses, name='total_loss')

                        # Generate summaries for the losses and get corresponding op
                        loss_averages_op = _add_loss_summaries(total_loss, losses, flags)

                        # Reuse variables for the next worker.
                        tf.get_variable_scope().reuse_variables()

                        # get summaries
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the current data batch
                        with tf.control_dependencies([loss_averages_op]):
                            grads = opt.compute_gradients(total_loss)

                        # Accumulate gradients across all workers.
                        worker_grads.append(grads)

                        # Accumulate extra non-standard operations across workers
                        worker_ops.append(n_net.get_misc_ops())

        # Average gradients over workers.
        avg_gradients = _average_gradients(worker_grads)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(avg_gradients, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            hyper_params['moving_average_decay'], global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Gather all training related ops into a single one.
        with tf.control_dependencies([apply_gradient_op, variables_averages_op, tf.group(*worker_ops)]):
            train_op = tf.no_op(name='train')

        # class _LoggerHook(tf.train.SessionRunHook):
        #     """Logs loss and runtime."""
        #
        #     def begin(self):
        #         self._step = -1
        #         self._start_time = time.time()
        #
        #     def before_run(self, run_context):
        #         self._step += 1
        #         return tf.train.SessionRunArgs(total_loss)  # Asks for loss value.
        #
        #     def after_run(self, run_context, run_values):
        #         if self._step % flags.log_frequency == 0:
        #             current_time = time.time()
        #             duration = current_time - self._start_time
        #             self._start_time = current_time
        #
        #             loss_value = run_values.results
        #             examples_per_sec = flags.log_frequency * flags.batch_size * num_GPUS / duration
        #             sec_per_batch = float(duration / flags.log_frequency)
        #             elapsed_epochs = num_GPUS * self._step * flags.batch_size / flags.NUM_EXAMPLES_PER_EPOCH
        #             format_str = ('%s: step = %d, epoch = %2.2e, loss = %.2f (%.1f examples/sec; %.3f '
        #                         'sec/batch)')
        #             print (format_str % (datetime.now(), self._step, elapsed_epochs, loss_value,
        #                                examples_per_sec, sec_per_batch))


        # Config file for tf.Session()
        config = tf.ConfigProto(allow_soft_placement=flags.allow_soft_placement,
                                log_device_placement=flags.log_device_placement)

        logHook = _LoggerHook(flags, total_loss, num_GPUS)

        # Start Training Session
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=flags.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=flags.max_steps), tf.train.NanTensorHook(total_loss),logHook],
                config=config) as mon_sess:
                while not mon_sess.should_stop():
                    mon_sess.run(train_op)


def set_flags(checkpt_dir, batch_size=64, data_dir=None):
    """
    Sets flags that could change from one run to the next
    :param checkpt_dir:
    :param batch_size:
    :param data_dir:
    :return:
    """
    tf.app.flags.DEFINE_string('train_dir', checkpt_dir, """Directory where to write event logs and checkpoint.""")
    tf.app.flags.DEFINE_integer('batch_size', batch_size, """Number of images to process in a batch.""")
    tf.app.flags.DEFINE_string('data_dir', data_dir,"""Directory where data tfrecords is located""")