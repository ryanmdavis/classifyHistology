import tensorflow as tf
from classifyHistology.train_net import vars_phs_consts_metrics as vars

# define layer wrappers. Convolution layer and max pooling layer
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

# residual learning module. Uses pre-activation of weighting with BN and ReLU
# SEe here about update ops: https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
def resModule(x1, weights, biases, layer_num, stride=1, is_training=False):
    # first variable layer
    with tf.name_scope('sub_rm') as scope:
        wt_key,b_key,wt_var,b_var=vars.getLayerVarNames(layer_num)

        x2 = tf.layers.batch_normalization(x1, training=is_training, axis=3)
        x2 = tf.nn.relu(x2)
        x2 = tf.nn.conv2d(x2, weights[wt_key], strides=[1, stride, stride, 1], padding='SAME')
        x2 = tf.nn.bias_add(x2, biases[b_key])
    
        # second variable layer
        wt_key,b_key,wt_var,b_var=vars.getLayerVarNames(layer_num+1)
        x2 = tf.layers.batch_normalization(x2, training=is_training, axis=3)
        x2 = tf.nn.relu(x2)
        x2 = tf.nn.conv2d(x2, weights[wt_key], strides=[1, stride, stride, 1], padding='SAME')
        x2 = tf.nn.bias_add(x2, biases[b_key])
    
        return x2
# define the convolutional network:
def convNet1(x, weights, biases, th):  

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3, k=2)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
       
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# This is the same as comvNet1 above, but it has a dropout layer for regularization 
def convNet2(x, weights, biases, th, is_training=False):  

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)
    conv1_drop = tf.contrib.layers.dropout(conv1,is_training=is_training,keep_prob=th['dropout_keep_prob'])

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1_drop, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)
    conv2_drop = tf.contrib.layers.dropout(conv2,is_training=is_training,keep_prob=th['dropout_keep_prob'])

    conv3 = conv2d(conv2_drop, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3, k=2)
    conv3_drop = tf.contrib.layers.dropout(conv3,is_training=is_training,keep_prob=th['dropout_keep_prob'])


    # Fully connected layer
    # Reshape conv3 output to fit fully connected layer input
    fc1 = tf.reshape(conv3_drop, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    
    # dropout layer
    #fc1_drop = tf.contrib.layers.dropout(fc1, th['dropout_keep_prob'], is_training=is_training, scope='dropout1')
    fc1_drop = tf.contrib.layers.dropout(fc1,is_training=is_training,keep_prob=th['dropout_keep_prob'])
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1_drop, weights['out']), biases['out'])
    return out

# This is the same as comvNet1 above, but it has a dropout layer for regularization 
def convNet3(x, weights, biases, th, is_training=False):  

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)
    conv1_drop = tf.contrib.layers.dropout(conv1,is_training=is_training,keep_prob=th['dropout_keep_prob'])

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1_drop, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)
    conv2_drop = tf.contrib.layers.dropout(conv2,is_training=is_training,keep_prob=th['dropout_keep_prob'])

    conv3 = conv2d(conv2_drop, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3_drop = tf.contrib.layers.dropout(conv3,is_training=is_training,keep_prob=th['dropout_keep_prob'])

    # Reshape conv3 output to fit fully connected layer input
    #!!!!! Small possibility that I broke it here when trying to make conv4
    fc1 = tf.reshape(conv3_drop, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    
    # dropout layer
    #fc1_drop = tf.contrib.layers.dropout(fc1, th['dropout_keep_prob'], is_training=is_training, scope='dropout1')
    fc1_drop = tf.contrib.layers.dropout(fc1,is_training=is_training,keep_prob=th['dropout_keep_prob'])
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1_drop, weights['out']), biases['out'])
    return out



##############
# This is the same as comvNet1 above, but it has a dropout layer for regularization 
def convNet4(x, weights, biases, th, is_training=False):  

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)
    conv1_drop = tf.contrib.layers.dropout(conv1,is_training=is_training,keep_prob=th['dropout_keep_prob'])

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1_drop, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)
    conv2_drop = tf.contrib.layers.dropout(conv2,is_training=is_training,keep_prob=th['dropout_keep_prob'])

    conv3 = conv2d(conv2_drop, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    var_layers_in = tf.contrib.layers.dropout(conv3,is_training=is_training,keep_prob=th['dropout_keep_prob'])

    # Add in the variable number of layers here
    # Need to add a new function like conv2d, except doesn't pass tensor thru nonlinearity.
    # double check that this is true, that residual goes in before nonlinearity
    with tf.name_scope('residual_modules') as scope:
        stride=1
        for layer_num in range(0,th['convNet4_num_layers'],2):
            var_layers_out = resModule(var_layers_in, weights, biases, layer_num, is_training=is_training)
            if th['convNet4_residual']:
                var_layers_in = tf.math.add(var_layers_in,var_layers_out)
            else:
                var_layers_in = var_layers_out
        fc_in = tf.nn.relu(var_layers_in)
    # Fully connected layer
    # Reshape conv3 output to fit fully connected layer input
    fc1 = tf.reshape(fc_in, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    
    # dropout layer
    #fc1_drop = tf.contrib.layers.dropout(fc1, th['dropout_keep_prob'], is_training=is_training, scope='dropout1')
    fc1_drop = tf.contrib.layers.dropout(fc1,is_training=is_training,keep_prob=th['dropout_keep_prob'])
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1_drop, weights['out']), biases['out'])
    return out