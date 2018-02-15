"""
Created on 12/15/17.
@author: Numan Laanait.
email: laanaitn@ornl.gov
"""

import tensorflow as tf
import argparse
import sys
import os
import horovod.tensorflow as hvd

sys.path.append('../')
from stemdl import runtime
from stemdl import io_utils


# NOTE because of summitdev/container problems, we can't pass any flags
# whatsoever, so we have to hard code this path
JSON_FLAGS = '../json/regress_flags_perangles.json'


def main(argv):
    # initiate horovod
    hvd.init()

    params = io_utils.get_dict_from_json(JSON_FLAGS)

    checkpt_dir = args.checkpt_dir[0]
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
    runtime.set_flags(checkpt_dir, eval_dir, args.batch_size, args.data_path[0])

    # load network config file and hyper_parameters
    network_config = io_utils.load_json_network_config(args.network_config[0])
    hyper_params = io_utils.load_json_hyper_params(args.hyper_params[0])

    # Create logfile to redirect all print statements
    # sys.stdout = open(args.mode[0] + '.log', mode='w')

    tf.app.flags.DEFINE_string('CPU_ID', '/cpu:' + str(args.cpu_id), """CPU_ID""")

    # train or evaluate
    if args.mode[0] == 'train':
        runtime.train_horovod(network_config, hyper_params, args.data_path[0], tf.app.flags.FLAGS, num_GPUS=args.num_gpus)
    if args.mode[0] == 'eval':
        runtime.eval(network_config, hyper_params, args.data_path[0], tf.app.flags.FLAGS, num_GPUS=args.num_gpus)

if __name__ == '__main__':
    main(sys.argv)
