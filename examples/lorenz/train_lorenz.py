#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append("../../src")
import os
import datetime
import pandas as pd
import numpy as np
from example_lorenz import get_lorenz_data
from sindy_utils import library_size
from training import train_network
import tensorflow as tf


# # Generate data

# In[ ]:


# generate training, validation, testing data
noise_strength = 1e-6
training_data = get_lorenz_data(1024, noise_strength=noise_strength)
validation_data = get_lorenz_data(20, noise_strength=noise_strength)


# # Set up model and training parameters

# In[ ]:


params = {}

params['input_dim'] = 128
params['latent_dim'] = 3
params['model_order'] = 1
params['poly_order'] = 3
params['include_sine'] = False
params['library_dim'] = library_size(params['latent_dim'], params['poly_order'], params['include_sine'], True)

# sequential thresholding parameters
params['sequential_thresholding'] = True
params['coefficient_threshold'] = 0.1
params['threshold_frequency'] = 500
params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
params['coefficient_initialization'] = 'constant'

# loss function weighting
params['loss_weight_decoder'] = 1.0
params['loss_weight_sindy_z'] = 0.0
params['loss_weight_sindy_x'] = 1e-4
params['loss_weight_sindy_regularization'] = 1e-5

params['activation'] = 'sigmoid'
params['widths'] = [64,32]

# training parameters
params['epoch_size'] = training_data['x'].shape[0]
params['batch_size'] = 1024
params['learning_rate'] = 1e-3

params['data_path'] = os.getcwd() + '/'
params['print_progress'] = True
params['print_frequency'] = 100

# training time cutoffs
params['max_epochs'] = 5001
params['refinement_epochs'] = 1001


# # Run training experiments

# In[ ]:


num_experiments = 1
df = pd.DataFrame()
for i in range(num_experiments):
    print('EXPERIMENT %d' % i)

    params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))

    params['save_name'] = 'lorenz_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    tf.reset_default_graph()

    results_dict = train_network(training_data, validation_data, params)
    df = df.append({**results_dict, **params}, ignore_index=True)

df.to_pickle('experiment_results_' + datetime.datetime.now().strftime("%Y%m%d%H%M") + '.pkl')

