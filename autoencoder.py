import tensorflow as tf


def full_network(params):
    """
    Define the full network architecture.

    Arguments:
        params - Dictionary object containing the parameters that specify the training.
        See README file for a description of the parameters.

    Returns:
        network - Dictionary containing the tensorflow objects that make up the network.
    """
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
        sindy_coefficients = tf.get_variable('sindy_coefficients', shape=[l,d], initializer=tf.contrib.layers.xavier_initializer())
    elif params['coefficient_initialization'] == 'specified':
        sindy_coefficients = tf.get_variable('sindy_coefficients', initializer=params['init_coefficients'])
    elif params['coefficient_initialization'] == 'constant':
        sindy_coefficients = tf.get_variable('sindy_coefficients', shape=[l,d], initializer=tf.constant_initializer(1.0))
    elif params['coefficient_initialization'] == 'normal':
        sindy_coefficients = tf.get_variable('sindy_coefficients', shape=[l,d], initializer=tf.initializers.random_normal())
    
    if params['sequential_thresholding']:
        coefficient_mask = tf.placeholder(tf.float32, shape=[l,d], name='coefficient_mask')
        sindy_predict = tf.matmul(Theta, coefficient_mask*sindy_coefficients)
        network['coefficient_mask'] = coefficient_mask
    else:
        sindy_predict = tf.matmul(Theta, sindy_coefficients)

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
    network['sindy_coefficients'] = sindy_coefficients

    if model_order == 1:
        network['dz_predict'] = sindy_predict
    else:
        network['ddz'] = ddz
        network['ddz_predict'] = sindy_predict
        network['ddx'] = ddx
        network['ddx_decode'] = ddx_decode

    return network


def define_loss(network, params):
    """
    Create the loss functions.

    Arguments:
        network - Dictionary object containing the elements of the network architecture.
        This will be the output of the full_network() function.
    """
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
    sindy_coefficients = params['coefficient_mask']*network['sindy_coefficients']

    losses = {}
    losses['decoder'] = tf.reduce_mean((x - x_decode)**2)
    if params['model_order'] == 1:
        losses['sindy_z'] = tf.reduce_mean((dz - dz_predict)**2)
        losses['sindy_x'] = tf.reduce_mean((dx - dx_decode)**2)
    else:
        losses['sindy_z'] = tf.reduce_mean((ddz - ddz_predict)**2)
        losses['sindy_x'] = tf.reduce_mean((ddx - ddx_decode)**2)
    losses['sindy_regularization'] = tf.reduce_mean(tf.abs(sindy_coefficients))
    loss = params['loss_weight_decoder'] * losses['decoder'] \
           + params['loss_weight_sindy_z'] * losses['sindy_z'] \
           + params['loss_weight_sindy_x'] * losses['sindy_x'] \
           + params['loss_weight_sindy_regularization'] * losses['sindy_regularization']

    loss_refinement = params['loss_weight_decoder'] * losses['decoder'] \
                      + params['loss_weight_sindy_z'] * losses['sindy_z'] \
                      + params['loss_weight_sindy_x'] * losses['sindy_x']

    return loss, losses, loss_refinement


def linear_autoencoder(x, N, d):
    # z,encoder_weights,encoder_biases = encoder(x, N, d, [], None, 'encoder')
    # x_decode,decoder_weights,decoder_biases = decoder(z, N, d, [], None, 'decoder')
    z,encoder_weights,encoder_biases = build_network_layers(x, N, d, [], None, 'encoder')
    x_decode,decoder_weights,decoder_biases = build_network_layers(z, d, N, [], None, 'decoder')

    return z, x_decode, encoder_weights, encoder_biases,decoder_weights,decoder_biases


def nonlinear_autoencoder(x, N, d, widths, activation='elu'):
    """
    Construct a nonlinear autoencoder.

    Arguments:

    Returns:
        z -
        x_decode -
        encoder_weights - List of tensorflow arrays containing the encoder weights
        encoder_biases - List of tensorflow arrays containing the encoder biases
        decoder_weights - List of tensorflow arrays containing the decoder weights
        decoder_biases - List of tensorflow arrays containing the decoder biases
    """
    if activation == 'relu':
        activation_function = tf.nn.relu
    elif activation == 'elu':
        activation_function = tf.nn.elu
    elif activation == 'sigmoid':
        activation_function = tf.sigmoid
    else:
        raise ValueError('invalid activation function')
    # z,encoder_weights,encoder_biases = encoder(x, N, d, widths, activation_function, 'encoder')
    # x_decode,decoder_weights,decoder_biases = decoder(z, N, d, widths[::-1], activation_function, 'decoder')
    z,encoder_weights,encoder_biases = build_network_layers(x, N, d, widths, activation_function, 'encoder')
    x_decode,decoder_weights,decoder_biases = build_network_layers(z, d, N, widths[::-1], activation_function, 'decoder')

    return z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases


def build_network_layers(input, input_dim, output_dim, widths, activation, name):
    """
    Construct one portion of the network (either encoder or decoder).

    Arguments:
        input - 2D tensorflow array, input to the network (shape is [?,input_dim])
        input_dim - Integer, number of state variables in the input to the first layer
        output_dim - Integer, number of state variables to output from the final layer
        widths - List of integers representing how many units are in each network layer
        activation - Tensorflow function to be used as the activation function at each layer
        name - String, prefix to be used in naming the tensorflow variables

    Returns:
        input - Tensorflow array, output of the network layers (shape is [?,output_dim])
        weights - List of tensorflow arrays containing the network weights
        biases - List of tensorflow arrays containing the network biases
    """
    weights = []
    biases = []
    last_width=input_dim
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
    W = tf.get_variable(name+'_W'+str(len(widths)), shape=[last_width,output_dim],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name+'_b'+str(len(widths)), shape=[output_dim],
        initializer=tf.constant_initializer(0.0))
    input = tf.matmul(input,W) + b
    weights.append(W)
    biases.append(b)
    return input, weights, biases


# def encoder(input, N, d, widths, activation, name):
#     """
#     Construct the encoder.

#     Arguments:
#         input - 2D tensorflow array, input to the network (shape is [?,N])
#         N - Integer, number of state variables in the original data space
#         d - Integer, number of state variables in the decoder space
#         widths - List of integers representing how many units are in each network layer
#         activation - Tensorflow function to be used as the activation function at each layer
#         name - String, prefix to be used in naming the tensorflow variables

#     Returns:
#         input - Tensorflow array, output of the encoder (shape is [?,d])
#         weights - List of tensorflow arrays containing the network weights
#         biases - List of tensorflow arrays containing the network biases
#     """
#     weights = []
#     biases = []
#     last_width=N
#     for i,n_units in enumerate(widths):
#         W = tf.get_variable(name+'_W'+str(i), shape=[last_width,n_units],
#             initializer=tf.contrib.layers.xavier_initializer())
#         b = tf.get_variable(name+'_b'+str(i), shape=[n_units],
#             initializer=tf.constant_initializer(0.0))
#         input = tf.matmul(input, W) + b
#         if activation is not None:
#             input = activation(input)
#         last_width = n_units
#         weights.append(W)
#         biases.append(b)
#     W = tf.get_variable(name+'_W'+str(len(widths)), shape=[last_width,d],
#         initializer=tf.contrib.layers.xavier_initializer())
#     b = tf.get_variable(name+'_b'+str(len(widths)), shape=[d],
#         initializer=tf.constant_initializer(0.0))
#     input = tf.matmul(input,W) + b
#     weights.append(W)
#     biases.append(b)
#     return input, weights, biases


# def decoder(input, N, d, widths, activation, name):
#     """
#     Construct the decoder.

#     Arguments:
#         input - 2D tensorflow array, input to the network (shape is [?,d])
#         N - Integer, number of state variables in the original data space
#         d - Integer, number of state variables in the decoder space
#         widths - List of integers representing how many units are in each network layer
#         activation - Tensorflow function to be used as the activation function at each layer
#         name - String, prefix to be used in naming the tensorflow variables

#     Returns:
#         input - Tensorflow array, output of the decoder (shape is [?,N])
#         weights - List of tensorflow arrays containing the network weights
#         biases - List of tensorflow arrays containing the network biases
#     """
#     weights = []
#     biases = []
#     last_width=d
#     for i,n_units in enumerate(widths):
#         W = tf.get_variable(name+'_W'+str(i), shape=[last_width,n_units],
#             initializer=tf.contrib.layers.xavier_initializer())
#         b = tf.get_variable(name+'_b'+str(i), shape=[n_units],
#             initializer=tf.constant_initializer(0.0))
#         input = tf.matmul(input, W) + b
#         if activation is not None:
#             input = activation(input)
#         last_width = n_units
#         weights.append(W)
#         biases.append(b)
#     W = tf.get_variable(name+'_W'+str(len(widths)), shape=[last_width,N],
#         initializer=tf.contrib.layers.xavier_initializer())
#     b = tf.get_variable(name+'_b'+str(len(widths)), shape=[N],
#         initializer=tf.constant_initializer(0.0))
#     input = tf.matmul(input,W) + b
#     weights.append(W)
#     biases.append(b)
#     return input, weights, biases


def sindy_library_tf(z, d, poly_order, include_sine=False):
    """
    Build the SINDy library.

    Arguments:
        z - 2D tensorflow array of the snapshots on which to build the library. Shape is number of
        time points by the number of state variables.
        d - Integer, number of state variable in z.
        poly_order - Integer, polynomial order to which to build the library. Max value is 5.
        include_sine - Boolean, whether or not to include sine terms in the library. Default False.

    Returns:
        2D tensorflow array containing the constructed library. Shape is number of time points by
        number of library functions. The number of library functions is determined by the number
        of state variables of the input, the polynomial order, and whether or not sines are included.
    """
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
    """
    Build the SINDy library for a second order system. This is essentially the same as for a first
    order system, but library terms are also built for the derivatives.
    """
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
    """
    Compute the first order time derivatives by propagating through the network.

    Arguments:
        input - 2D tensorflow array, input to the network. Dimensions are number of time points
        by number of state variables.
        dx - First order time derivatives of the input to the network.
        weights - List of tensorflow arrays containing the network weights
        biases - List of tensorflow arrays containing the network biases
        activation - String specifying which activation function to use. Options are
        'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
        or linear.

    Returns:
        dz - Tensorflow array, first order time derivatives of the network output.
    """
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
    """
    Compute the first and second order time derivatives by propagating through the network.

    Arguments:
        input - 2D tensorflow array, input to the network. Dimensions are number of time points
        by number of state variables.
        dx - First order time derivatives of the input to the network.
        ddx - Second order time derivatives of the input to the network.
        weights - List of tensorflow arrays containing the network weights
        biases - List of tensorflow arrays containing the network biases
        activation - String specifying which activation function to use. Options are
        'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
        or linear.

    Returns:
        dz - Tensorflow array, first order time derivatives of the network output.
        ddz - Tensorflow array, second order time derivatives of the network output.
    """
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
