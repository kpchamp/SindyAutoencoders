import os
import datetime
import pandas as pd
import numpy as np
from example_reactiondiffusion import get_rd_data
from sindy_utils import library_size
from training import train_network
import tensorflow as tf


# SET UP PARAMETERS
params = {}

# generate training, validation, testing data
training_data, val_data, test_data = get_rd_data()

params['N'] = training_data['x'].size*training_data['y'].size
params['d'] = 2
params['model_order'] = 1
params['poly_order'] = 3
params['include_sine'] = True
params['l'] = library_size(params['d'], params['poly_order'], params['include_sine'], True)

# set up sequential thresholding
params['sequential_thresholding'] = True
params['coefficient_threshold'] = 0.1
params['threshold_frequency'] = 500
params['coefficient_mask'] = np.ones((params['l'], params['d']))
params['coefficient_initialization'] = 'constant'

# define loss weights
params['loss_weight_decoder'] = 1.0
params['loss_weight_sindy_z'] = 0.01
params['loss_weight_sindy_x'] = 0.5
params['loss_weight_sindy_regularization'] = 0.1

params['activation'] = 'sigmoid'
params['widths'] = [256]

# training parameters
params['epoch_size'] = training_data['t'].size
params['batch_size'] = 1024
params['learning_rate'] = 1e-3

params['data_path'] = os.getcwd() + '/'
params['print_progress'] = True
params['print_frequency'] = 100

# training time cutoffs
params['max_epochs'] = 3001
params['refinement_epochs'] = 1001

num_experiments = 10
df = pd.DataFrame()
for i in range(num_experiments):
    print('EXPERIMENT %d' % i)

    params['coefficient_mask'] = np.ones((params['l'], params['d']))

    params['save_name'] = 'rd_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    tf.reset_default_graph()

    num_epochs, x_norm, dx_norm, decoder_loss,decoder_sindy_loss, sindy_regularization = train_network(training_data, val_data, params)
    results_dict = {'num_epochs': num_epochs, 'x_norm': x_norm,
                    'dx_norm': dx_norm, 'decoder_loss': decoder_loss,
                    'decoder_sindy_loss': decoder_sindy_loss, 'sindy_regularization': sindy_regularization}
    df = df.append({**results_dict, **params}, ignore_index=True)
        
df.to_pickle('experiment_results_' + datetime.datetime.now().strftime("%Y%m%d%H%M") + '.pkl')
