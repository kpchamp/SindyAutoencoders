import tensorflow as tf


def full_network(params):
    N = params['N']
    d = params['d']
    activation = params['activation']
    poly_order = params['poly_order']
    if 'include_sine' in params.keys():
        include_sine = params['include_sine']
    else:
        include_sine = False
    l = params['l']
    model_order = params['model_order']

    network = {}

    x = tf.placeholder(tf.float32, shape=[None, N], name='x')
    dx = tf.placeholder(tf.float32, shape=[None, N], name='dx')
    if model_order == 2:
        ddx = tf.placeholder(tf.float32, shape=[None, N], name='ddx')

    if activation == 'linear':
        z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases = linear_autoencoder(x, N, d)
    else:
        z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases = nonlinear_autoencoder(x, N, d, params['widths'], activation=activation)
    
    if model_order == 1:
        dz = z_derivative(x, dx, encoder_weights, encoder_biases, activation=activation)
        Theta = sindy_library_tf(z, d, poly_order, include_sine)
    else:
        dz,ddz = z_derivative_order2(x, dx, ddx, encoder_weights, encoder_biases, activation=activation)
        Theta = sindy_library_tf_order2(z, dz, d, poly_order, include_sine)

    if params['coefficient_initialization'] == 'xavier':
        Xi = tf.get_variable('Xi', shape=[l,d], initializer=tf.contrib.layers.xavier_initializer())
    elif params['coefficient_initialization'] == 'specified':
        Xi = tf.get_variable('Xi', initializer=params['init_coefficients'])
    elif params['coefficient_initialization'] == 'constant':
        Xi = tf.get_variable('Xi', shape=[l,d], initializer=tf.constant_initializer(1.0))
    elif params['coefficient_initialization'] == 'normal':
        Xi = tf.get_variable('Xi', shape=[l,d], initializer=tf.initializers.random_normal())
    
    if params['sequential_thresholding']:
        coefficient_mask = tf.placeholder(tf.float32, shape=[l,d], name='coefficient_mask')
        sindy_predict = tf.matmul(Theta, coefficient_mask*Xi)
        network['coefficient_mask'] = coefficient_mask
    else:
        sindy_predict = tf.matmul(Theta, Xi)

    if model_order == 1:
        dx_decode = z_derivative(z, sindy_predict, decoder_weights, decoder_biases, activation=activation)
    else:
        dx_decode,ddx_decode = z_derivative_order2(z, dz, sindy_predict, decoder_weights, decoder_biases,
                                             activation=activation)

    network['x'] = x
    network['dx'] = dx
    network['z'] = z
    network['dz'] = dz
    network['x_decode'] = x_decode
    network['dx_decode'] = dx_decode
    network['encoder_weights'] = encoder_weights
    network['encoder_biases'] = encoder_biases
    network['decoder_weights'] = decoder_weights
    network['decoder_biases'] = decoder_biases
    network['Theta'] = Theta
    network['Xi'] = Xi

    if model_order == 1:
        network['dz_predict'] = sindy_predict
    else:
        network['ddz'] = ddz
        network['ddz_predict'] = sindy_predict
        network['ddx'] = ddx
        network['ddx_decode'] = ddx_decode

    return network


def define_loss(network, params):
    x = network['x']
    x_decode = network['x_decode']
    if params['model_order'] == 1:
        dz = network['dz']
        dz_predict = network['dz_predict']
        dx = network['dx']
        dx_decode = network['dx_decode']
    else:
        ddz = network['ddz']
        ddz_predict = network['ddz_predict']
        ddx = network['ddx']
        ddx_decode = network['ddx_decode']
    Xi = network['Xi']

    losses = {}
    losses['decoder'] = tf.reduce_mean((x - x_decode)**2)
    if params['model_order'] == 1:
        losses['sindy_z'] = tf.reduce_mean((dz - dz_predict)**2)
        losses['sindy_x'] = tf.reduce_mean((dx - dx_decode)**2)
    else:
        losses['sindy_z'] = tf.reduce_mean((ddz - ddz_predict)**2)
        losses['sindy_x'] = tf.reduce_mean((ddx - ddx_decode)**2)
    losses['sindy_regularization'] = tf.reduce_mean(tf.abs(Xi))
    loss = params['loss_weight_decoder'] * losses['decoder'] \
           + params['loss_weight_sindy_z'] * losses['sindy_z'] \
           + params['loss_weight_sindy_x'] * losses['sindy_x'] \
           + params['loss_weight_sindy_regularization'] * losses['sindy_regularization']

    loss_refinement = params['loss_weight_decoder'] * losses['decoder'] \
                      + params['loss_weight_sindy_z'] * losses['sindy_z'] \
                      + params['loss_weight_sindy_x'] * losses['sindy_x']

    return loss, losses, loss_refinement


def linear_autoencoder(x, N, d):
    z,encoder_weights,encoder_biases = encoder(x, N, d, [], None, 'encoder')
    x_decode,decoder_weights,decoder_biases = decoder(z, N, d, [], None, 'decoder')

    return z, x_decode, encoder_weights, encoder_biases,decoder_weights,decoder_biases


def nonlinear_autoencoder(x, N, d, widths, activation='elu'):
    if activation == 'relu':
        activation_function = tf.nn.relu
    elif activation == 'elu':
        activation_function = tf.nn.elu
    elif activation == 'sigmoid':
        activation_function = tf.sigmoid
    else:
        raise ValueError('invalid activation function')
    z,encoder_weights,encoder_biases = encoder(x, N, d, widths, activation_function, 'encoder')
    x_decode,decoder_weights,decoder_biases = decoder(z, N, d, widths[::-1], activation_function, 'decoder')

    return z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases


def encoder(input, N, d, widths, activation, name, training_mode=False):
    weights = []
    biases = []
    last_width=N
    for i,n_units in enumerate(widths):
        W = tf.get_variable(name+'_W'+str(i), shape=[last_width,n_units],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name+'_b'+str(i), shape=[n_units],
            initializer=tf.constant_initializer(0.0))
        input = tf.matmul(input, W) + b
        if activation is not None:
            input = activation(input)
        last_width = n_units
        weights.append(W)
        biases.append(b)
    W = tf.get_variable(name+'_W'+str(len(widths)), shape=[last_width,d],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name+'_b'+str(len(widths)), shape=[d],
        initializer=tf.constant_initializer(0.0))
    input = tf.matmul(input,W) + b
    weights.append(W)
    biases.append(b)
    return input, weights, biases


def decoder(input, N, d, widths, activation, name, training_mode=False):
    weights = []
    biases = []
    last_width=d
    for i,n_units in enumerate(widths):
        W = tf.get_variable(name+'_W'+str(i), shape=[last_width,n_units],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name+'_b'+str(i), shape=[n_units],
            initializer=tf.constant_initializer(0.0))
        input = tf.matmul(input, W) + b
        if activation is not None:
            input = activation(input)
        last_width = n_units
        weights.append(W)
        biases.append(b)
    W = tf.get_variable(name+'_W'+str(len(widths)), shape=[last_width,N],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name+'_b'+str(len(widths)), shape=[N],
        initializer=tf.constant_initializer(0.0))
    input = tf.matmul(input,W) + b
    weights.append(W)
    biases.append(b)
    return input, weights, biases


def sindy_library_tf(z, d, poly_order, include_sine=False):
    library = [tf.ones(tf.shape(z)[0])]

    for i in range(d):
        library.append(z[:,i])

    if poly_order > 1:
        for i in range(d):
            for j in range(i,d):
                library.append(tf.multiply(z[:,i], z[:,j]))

    if poly_order > 2:
        for i in range(d):
            for j in range(i,d):
                for k in range(j,d):
                    library.append(z[:,i]*z[:,j]*z[:,k])

    if poly_order > 3:
        for i in range(d):
            for j in range(i,d):
                for k in range(j,d):
                    for p in range(k,d):
                        library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p])

    if poly_order > 4:
        for i in range(d):
            for j in range(i,d):
                for k in range(j,d):
                    for p in range(k,d):
                        for q in range(p,d):
                            library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p]*z[:,q])

    if include_sine:
        for i in range(d):
            library.append(tf.sin(z[:,i]))

    return tf.stack(library, axis=1)


def sindy_library_tf_order2(z, dz, d, poly_order, include_sine=False):
    library = [tf.ones(tf.shape(z)[0])]

    z_combined = tf.concat([z, dz], 1)

    for i in range(2*d):
        library.append(z_combined[:,i])

    if poly_order > 1:
        for i in range(2*d):
            for j in range(i,2*d):
                library.append(tf.multiply(z_combined[:,i], z_combined[:,j]))

    if poly_order > 2:
        for i in range(2*d):
            for j in range(i,2*d):
                for k in range(j,2*d):
                    library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k])

    if poly_order > 3:
        for i in range(2*d):
            for j in range(i,2*d):
                for k in range(j,2*d):
                    for p in range(k,2*d):
                        library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k]*z_combined[:,p])

    if poly_order > 4:
        for i in range(2*d):
            for j in range(i,2*d):
                for k in range(j,2*d):
                    for p in range(k,2*d):
                        for q in range(p,2*d):
                            library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k]*z_combined[:,p]*z_combined[:,q])

    if include_sine:
        for i in range(2*d):
            library.append(tf.sin(z_combined[:,i]))

    return tf.stack(library, axis=1)


def z_derivative(input, dx, weights, biases, activation='elu'):
    dz = dx
    if activation == 'elu':
        for i in range(len(weights)-1):
            input = tf.matmul(input, weights[i]) + biases[i]
            dz = tf.multiply(tf.minimum(tf.exp(input),1.0),
                                  tf.matmul(dz, weights[i]))
            input = tf.nn.elu(input)
        dz = tf.matmul(dz, weights[-1])
    elif activation == 'relu':
        for i in range(len(weights)-1):
            input = tf.matmul(input, weights[i]) + biases[i]
            dz = tf.multiply(tf.to_float(input>0), tf.matmul(dz, weights[i]))
            input = tf.nn.relu(input)
        dz = tf.matmul(dz, weights[-1])
    elif activation == 'sigmoid':
        for i in range(len(weights)-1):
            input = tf.matmul(input, weights[i]) + biases[i]
            input = tf.sigmoid(input)
            dz = tf.multiply(tf.multiply(input, 1-input), tf.matmul(dz, weights[i]))
        dz = tf.matmul(dz, weights[-1])
    else:
        for i in range(len(weights)-1):
            dz = tf.matmul(dz, weights[i])
        dz = tf.matmul(dz, weights[-1])
    return dz


def z_derivative_order2(input, dx, ddx, weights, biases, activation='elu'):
    dz = dx
    ddz = ddx
    if activation == 'elu':
        for i in range(len(weights)-1):
            input = tf.matmul(input, weights[i]) + biases[i]
            dz_prev = tf.matmul(dz, weights[i])
            elu_derivative = tf.minimum(tf.exp(input),1.0)
            elu_derivative2 = tf.multiply(tf.exp(input), tf.to_float(input<0))
            dz = tf.multiply(elu_derivative, dz_prev)
            ddz = tf.multiply(elu_derivative2, tf.square(dz_prev)) \
                  + tf.multiply(elu_derivative, tf.matmul(ddz, weights[i]))
            input = tf.nn.elu(input)
        dz = tf.matmul(dz, weights[-1])
        ddz = tf.matmul(ddz, weights[-1])
    elif activation == 'relu':
        # NOTE: currently having trouble assessing accuracy of 2nd derivative due to discontinuity
        for i in range(len(weights)-1):
            input = tf.matmul(input, weights[i]) + biases[i]
            relu_derivative = tf.to_float(input>0)
            dz = tf.multiply(relu_derivative, tf.matmul(dz, weights[i]))
            ddz = tf.multiply(relu_derivative, tf.matmul(ddz, weights[i]))
            input = tf.nn.relu(input)
        dz = tf.matmul(dz, weights[-1])
        ddz = tf.matmul(ddz, weights[-1])
    elif activation == 'sigmoid':
        for i in range(len(weights)-1):
            input = tf.matmul(input, weights[i]) + biases[i]
            input = tf.sigmoid(input)
            dz_prev = tf.matmul(dz, weights[i])
            sigmoid_derivative = tf.multiply(input, 1-input)
            sigmoid_derivative2 = tf.multiply(sigmoid_derivative, 1 - 2*input)
            dz = tf.multiply(sigmoid_derivative, dz_prev)
            ddz = tf.multiply(sigmoid_derivative2, tf.square(dz_prev)) \
                  + tf.multiply(sigmoid_derivative, tf.matmul(ddz, weights[i]))
        dz = tf.matmul(dz, weights[-1])
        ddz = tf.matmul(ddz, weights[-1])
    else:
        for i in range(len(weights)-1):
            dz = tf.matmul(dz, weights[i])
            ddz = tf.matmul(ddz, weights[i])
        dz = tf.matmul(dz, weights[-1])
        ddz = tf.matmul(ddz, weights[-1])
    return dz,ddz


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)
