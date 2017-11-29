"""
Created on 10/20/17.
@author: Numan Laanait.
email: laanaitn@ornl.gov
"""
import tensorflow as tf

with tf.device('/gpu:0'):
    var_1 = tf.constant([1,1])
    var_2 = tf.constant([2,2])
    var_3 = var_1*var_2

with tf.Session() as sess:
    prod = sess.run(var_3)
    print('Product is %s' % format(prod))