import numpy as np
import tensorflow as tf
import pickle
from autoencoder import full_network, define_loss


def train_network(training_data, val_data, params):
    # SET UP NETWORK
    autoencoder_network = full_network(params)
    loss, losses, loss_refinement = define_loss(autoencoder_network, params)
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    train_op_refinement = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_refinement)
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    validation_dict = create_feed_dictionary(val_data, params, idxs=None)

    x_norm = np.mean(val_data['u']**2)
    if params['model_order'] == 1:
        sindy_predict_norm = np.mean(val_data['du']**2)
    else:
        sindy_predict_norm = np.mean(val_data['ddu']**2)

    print('TRAINING')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(params['max_epochs']):
            for j in range(params['epoch_size']//params['batch_size']):
                batch_idxs = np.arange(j*params['batch_size'], (j+1)*params['batch_size'])
                train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
                sess.run(train_op, feed_dict=train_dict)
            
            if params['print_progress'] and (i % params['print_frequency'] == 0):
                print_progress(sess, i, loss, losses, train_dict, validation_dict, x_norm, sindy_predict_norm)

            if params['sequential_thresholding'] and (i % params['threshold_frequency'] == 0) and (i > 0):
                params['coefficient_mask'] = np.abs(sess.run(autoencoder_network['Xi'])) > params['coefficient_threshold']
                validation_dict['coefficient_mask:0'] = params['coefficient_mask']
                print('THRESHOLDING: %d active coefficients' % np.sum(params['coefficient_mask']))

        print('REFINEMENT')
        for i_refinement in range(params['refinement_epochs']):
            for j in range(params['epoch_size']//params['batch_size']):
                batch_idxs = np.arange(j*params['batch_size'], (j+1)*params['batch_size'])
                train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
                sess.run(train_op_refinement, feed_dict=train_dict)
            
            if params['print_progress'] and (i_refinement % params['print_frequency'] == 0):
                print_progress(sess, i_refinement, loss_refinement, losses, train_dict, validation_dict, x_norm, sindy_predict_norm)

        saver.save(sess, params['data_path'] + params['save_name'])
        pickle.dump(params, open(params['data_path'] + params['save_name'] + '_params.pkl', 'wb'))
        decoder_losses = sess.run((losses['decoder'], losses['sindy_x']), feed_dict=validation_dict)
        regularization_loss = sess.run(losses['sindy_regularization'], feed_dict=validation_dict)

        return i, x_norm, sindy_predict_norm, decoder_losses[0], decoder_losses[1], regularization_loss


def print_progress(sess, i, loss, losses, train_dict, validation_dict, x_norm, sindy_predict_norm):
    training_loss_vals = sess.run((loss,) + tuple(losses.values()), feed_dict=train_dict)
    validation_loss_vals = sess.run((loss,) + tuple(losses.values()), feed_dict=validation_dict)
    print("Epoch %d" % i)
    print("   training loss {0}, {1}".format(training_loss_vals[0],
                                             training_loss_vals[1:]))
    print("   validation loss {0}, {1}".format(validation_loss_vals[0],
                                               validation_loss_vals[1:]))
    decoder_losses = sess.run((losses['decoder'], losses['sindy_x']), feed_dict=validation_dict)
    loss_ratios = (decoder_losses[0]/x_norm, decoder_losses[1]/sindy_predict_norm)
    print("decoder loss ratio: %f, decoder SINDy loss  ratio: %f" % loss_ratios)


def create_feed_dictionary(data, params, idxs=None):
    if idxs is None:
        idxs = np.arange(data['u'].shape[0])
    feed_dict = {}
    feed_dict['x:0'] = data['u'][idxs]
    feed_dict['dx:0'] = data['du'][idxs]
    if params['model_order'] == 2:
        feed_dict['ddx:0'] = data['ddu'][idxs]
    if params['sequential_thresholding']:
        feed_dict['coefficient_mask:0'] = params['coefficient_mask']
    feed_dict['learning_rate:0'] = params['learning_rate']
    return feed_dict
