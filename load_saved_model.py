import os
import sys
import numpy as np
import pickle
from example_lorenz import get_lorenz_data
from example_pendulum import get_pendulum_data
from example_reactiondiffusion import get_rd_data
from training import create_feed_dictionary
from autoencoder import full_network, define_loss
import tensorflow as tf


data_path = os.getcwd() + '/'
example_problem = sys.argv[1]
save_name = data_path + sys.argv[2]

if example_problem == 'lorenz':
    test_data = get_lorenz_data(1, 1, 100)[2]
elif example_problem == 'pendulum':
    test_data = get_pendulum_data(1, 1, 50)[2]
else:
    test_data = get_rd_data()[2]

params = pickle.load(open(save_name + '_params.pkl', 'rb'))
test_dict = create_feed_dictionary(test_data, params)

autoencoder_network = full_network(params)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

run_tuple = ()
for key in autoencoder_network.keys():
    run_tuple += (autoencoder_network[key],)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, save_name)

    tf_results = sess.run(run_tuple, feed_dict=test_dict)

results = {}
for i,key in enumerate(autoencoder_network.keys()):
    results[key] = tf_results[i]
