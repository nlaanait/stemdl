"""
Created on 10/8/17.
@author: Numan Laanait.
email: laanaitn@ornl.gov
"""

from stemdl import inputs
from stemdl import runtime
from stemdl import io_utils
import tensorflow as tf
import argparse
import sys


"""
These FLAGS define variables for a particular TF workflow and are not expected to change.

"""
# Basic parameters describing the training run
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, """Whether to allow variable soft placement on the device-""" +\
                     """ This is needed for multi-gpu runs.""")
tf.app.flags.DEFINE_integer('log_frequency', 10, """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('save_frequency', 10, """How often to save summaries to disk""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,"""Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_epochs', 500,"""Number of Data Epochs to do training""")
tf.app.flags.DEFINE_integer('NUM_EXAMPLES_PER_EPOCH', 729000,"""Number of examples in training data.""")
tf.app.flags.DEFINE_string('worker_name', 'worker', """Name of gpu worker to append to each device ops, scope, etc...""")

# Basic parameters describing the data set.
tf.app.flags.DEFINE_integer('NUM_CLASSES', 3, """Number of classes in training/evaluation data.""")
tf.app.flags.DEFINE_integer('OUTPUT_DIM', 3, """Dimension of the Network's Output""")
tf.app.flags.DEFINE_integer('IMAGE_HEIGHT', 85, """IMAGE HEIGHT""")
tf.app.flags.DEFINE_integer('IMAGE_WIDTH', 120, """IMAGE WIDTH""")
tf.app.flags.DEFINE_integer('IMAGE_DEPTH', 1, """IMAGE DEPTH""")
tf.app.flags.DEFINE_integer('CROP_HEIGHT', 60, """CROP HEIGHT""")
tf.app.flags.DEFINE_integer('CROP_WIDTH', 80, """CROP WIDTH""")
tf.app.flags.DEFINE_boolean('IMAGE_FP16', False, """ Whether to use half-precision format for images.""")
FLAGS = tf.app.flags.FLAGS


def main(argv):  # pylint: disable=unused-argument
    parser = argparse.ArgumentParser(description='Setup and Run a Deep Neural Network.')
    parser.add_argument('--data_path', type=str, help='path to tfrecords file with images + labels.', nargs=1,
                        required=True)
    parser.add_argument('--checkpt_dir', type=str, help='path where to save directory with training data,'+\
                                                        'visualization, and TensorBoard events.', nargs=1,
                        required=True)
    parser.add_argument('--network_config', type=str, help='path to .json file with neural net architecture.', nargs=1,
                        required=True)
    parser.add_argument('--hyper_params', type=str, help='path to .json file with hyper-parameters.', nargs=1,
                        required=True)
    parser.add_argument('--mode', type=str, help="operation mode, must be 'train' or 'eval'.\nDefault 'train'. ",
                        nargs='*', default='train')
    parser.add_argument('--num_gpus', type=int, help='number of gpus to use during training.\nDefault 1.',
                        nargs='?', default=1)
    parser.add_argument('--batch_size', type=int, help='number of images per batch to propagate through the network.'+\
                        '\nPowers of 2 are processed more efficiently.\nDefault 64.', nargs='?', default=64)
    args = parser.parse_args()

    # Create the checkpoint directory
    if tf.gfile.Exists(args.checkpt_dir[0]):
        print('Directory "%s" exists already.\nReloading model from latest checkpoint.' %format(args.checkpt_dir[0]))
    else:
        tf.gfile.MakeDirs(args.checkpt_dir[0])

    # Set additional tf.app.flags
    runtime.set_flags(args.checkpt_dir[0], args.batch_size, args.data_path[0])

    # load network config file and hyper_parameters
    network_config = io_utils.load_json_network_config(args.network_config[0])
    hyper_params = io_utils.load_json_hyper_params(args.hyper_params[0])

    # train or evaluate
    if args.mode == 'train':
        runtime.train(network_config, hyper_params, args.data_path[0], tf.app.flags.FLAGS, num_GPUS=args.num_gpus)
    if args.mode == 'eval':
        pass

if __name__ == '__main__':
    main(sys.argv)
