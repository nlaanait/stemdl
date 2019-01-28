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
from stemdl import inputs




def benchmark_io(params, filetype= "tfrecord", num_batches=2):
    config = tf.ConfigProto(allow_soft_placement=params['allow_soft_placement'],
                            log_device_placement=params['log_device_placement'],
                            )
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.gpu_options.force_gpu_compatible = True
    config.intra_op_parallelism_threads = 1
    #config.inter_op_parallelism_threads = max(1, cpu_count()//6)
    config.inter_op_parallelism_threads = params['IO_threads'] 
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    # JIT causes gcc errors on dgx-dl and is built without on Summit.
    sess = tf.Session(config=config)
    
    # setup data stream
    with tf.device(params['CPU_ID']):
    #with tf.device('/gpu:%d' % hvd.local_rank()):
        with tf.name_scope('Input') as _:
            if filetype == "tfrecord":
                dset = inputs.DatasetTFRecords(params, dataset=params['dataset'], debug=False)
            elif filetype == "lmdb":
                dset = inputs.DatasetLMDB(params, dataset=params['dataset'], debug=False)
            images, labels = dset.minibatch()
            #print('rank %d: reading minibatch' % hvd.rank()) 
            # Staging images on host
            staging_op, (images, labels) = dset.stage([images, labels])
    #        images = tf.cast(images, tf.half)
    #        labels = tf.cast(labels, tf.half)
            #print('rank %d: staging op' % hvd.rank()) 

    with tf.device('/gpu:%d' % hvd.local_rank()):
        # Copy images from host to device
        gpucopy_op, (images, labels) = dset.stage([images, labels])
        images = tf.cast(images, tf.half)
        labels = tf.cast(labels, tf.half)
        IO_ops = [staging_op, gpucopy_op]
        #print('rank %d: gpu copy op' % hvd.rank()) 
        
    #Initialize variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Sync
    print('rank %d: Syncing horovod' % hvd.rank()) 
    sync_op = hvd.broadcast_global_variables(0)
    sess.run(sync_op)

    # prefill pipeline first
    print('rank %d: Prefilling I/O pipeline...' % hvd.rank())
    for i in range(len(IO_ops)):
        sess.run(IO_ops[:i + 1])
    #sess.run(IO_ops[0])
    #print('IO_ops: %s, %s' %(len(IO_ops), format(IO_ops)))
    t = time.time() 
    for i in range(num_batches):
        sess.run(IO_ops)
        #cbed, pot = sess.run([images, labels, IO_ops])[:2]
        #print("types %s" %format(cbed.dtype))
        #print("images and labels size: %s, %s" %(format(images.get_shape().as_list()), format(labels.get_shape().as_list())))
        print("rank %d: retrieved batch %d" %(hvd.rank(), i))
        #print("cbed: min %f, max %f, std %f" %(cbed.min(), cbed.max(), cbed.std()))
        #print("potential: min %f, max %f, std %f" %(pot.min(), pot.max(), pot.std()))
    #if hvd.rank() == 0:
    #    np.save("cbed_0.npy",cbed)
    #    np.save("pot_0.npy", pot)
    if images.dtype == 'float32':
        num_bytes = 4
    else:
        num_bytes = 2
    size = np.prod(np.array(images.get_shape().as_list())) * num_bytes * num_batches /1024e6
    print("rank %d: bandwidth= %2.3f GB/s" %(hvd.rank(), size/(time.time()-t)))
    cbed, pot = sess.run([images, labels, IO_ops])[:2]
    print("rank %d , cbed: min %f, max %f, dtype %s" %(hvd.rank(), cbed.min(), cbed.max(), cbed.dtype))
    print("rank %d , potential: min %f, max %f, dtype %s" %(hvd.rank(), pot.min(), pot.max(), pot.dtype))
