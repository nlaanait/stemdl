"""
Created on 12/15/17.
@author: Suhas Somnath
"""

import tensorflow as tf
import argparse
import sys
import os
import horovod.tensorflow as hvd

sys.path.append('../')
from stemdl import runtime
from stemdl import io_utils

io_utils.load_flags_from_json('regress_flags.json', tf.app.flags)

FLAGS = tf.app.flags.FLAGS


def main():
    # initiate horovod
    hvd.init()

    checkpt_dir = FLAGS.checkpt_dir
    # Also need a directory within the checkpoint dir for event files coming from eval
    eval_dir = os.path.join(checkpt_dir, '_eval')

    # Create directories
    if tf.gfile.Exists(checkpt_dir):
        print('Directory "%s" exists already.\nReloading model from latest checkpoint.' % format(checkpt_dir))
    elif tf.gfile.Exists(eval_dir):
        print('Removing "%s"' % format(eval_dir))
        tf.gfile.DeleteRecursively(eval_dir)
    else:
        tf.gfile.MakeDirs(checkpt_dir)
        tf.gfile.MakeDirs(eval_dir)

    # Set additional tf.app.flags
    runtime.set_flags(checkpt_dir, eval_dir, FLAGS.batch_size_2, FLAGS.data_path)
    print(FLAGS.train_dir, FLAGS.eval_dir, FLAGS.batch_size, FLAGS.data_dir)

    # load network config file and hyper_parameters
    network_config = io_utils.load_json_network_config(FLAGS.network_config)
    hyper_params = io_utils.load_json_hyper_params(FLAGS.hyper_params)

    # train or evaluate
    if FLAGS.mode == 'train':
        runtime.train_horovod(network_config, hyper_params, FLAGS.data_path, tf.app.flags.FLAGS, num_GPUS=FLAGS.num_gpus)
    if FLAGS.mode == 'eval':
        runtime.eval(network_config, hyper_params, FLAGS.data_path, tf.app.flags.FLAGS, num_GPUS=FLAGS.num_gpus)


if __name__ == '__main__':
    main()

