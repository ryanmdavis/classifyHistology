from classifyHistology.train_net import layers_and_net as net
import tensorflow as tf

def definePhVar(th):
    # Define all of the variables
    if th['n_input'][0] == 28:
        weights = {
            'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 
            'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
            'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
            'wd1': tf.get_variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
            'out': tf.get_variable('W6', shape=(128,th['n_classes']), initializer=tf.contrib.layers.xavier_initializer()), 
        }
        biases = {
            'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
            'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
            'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
        }
        x = tf.placeholder("float", [None, th['n_input'][0],th['n_input'][1],1])
    elif (th['n_input'][1] == 64) and (th['net'] is 'convNet2'):
        #input 32x64
        #conv 16x32
        #conv 8x16
        #conv 4x8
        #fully connected (4*8*128)x128 (second 128 is number of neurons in fully connected layer)
        #fc out: 128x2 (128 in past layer, 2 in next layer)
        weights = {
            'wc1': tf.get_variable('W0', shape=(3,3,3,32), initializer=tf.contrib.layers.xavier_initializer()), 
            'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
            'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
            'wd1': tf.get_variable('W3', shape=(4*8*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
            'out': tf.get_variable('W6', shape=(128,2), initializer=tf.contrib.layers.xavier_initializer()), 
        }
        biases = {
            'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
            'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
            'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('B4', shape=(th['n_classes']), initializer=tf.contrib.layers.xavier_initializer()),
        }
        x = tf.placeholder("float", [None, th['n_input'][0],th['n_input'][1],th['n_input'][2]])  
    elif th['net'] is 'convNet3':
        #input 16x64
        #conv 8x32
        #conv 4x16
        #conv 4x16 (omit the maxpool here)
        #fully connected (4*16*128)x128 (second 128 is number of neurons in fully connected layer)
        #fc out: 128x2 (128 in past layer, 2 in next layer)
        weights = {
            'wc1': tf.get_variable('W0', shape=(3,3,3,32), initializer=tf.contrib.layers.xavier_initializer()), 
            'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
            'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
            'wd1': tf.get_variable('W3', shape=(4*16*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
            'out': tf.get_variable('W6', shape=(128,2), initializer=tf.contrib.layers.xavier_initializer()), 
        }
        biases = {
            'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
            'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
            'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('B4', shape=(th['n_classes']), initializer=tf.contrib.layers.xavier_initializer()),
        }
        x = tf.placeholder("float", [None, th['n_input'][0],th['n_input'][1],th['n_input'][2]])          
    else:
        raise Exception
    # Define placeholders. Both placeholders are of type float
 
    y = tf.placeholder("float", [None, th['n_classes']])
    
    ph_is_training=tf.placeholder(tf.bool)
    
    learning_rate=tf.placeholder(tf.float32)
    
    return weights,biases,x,y,ph_is_training,learning_rate

def performanceMetrics(x,y,weights,biases,th,ph_is_training,ph_learning_rate): #,is_training):
    
    # get the network archetecture
    netFunc = getattr(net,th['net'])
    
    # define the networks, which outputs the logits
    pred = netFunc(x, weights, biases,th,is_training=ph_is_training) #,is_training)
    
    # Instructions for updating:
    # 
    # Future major versions of TensorFlow will allow gradients to flow
    # into the labels input on backprop by default.
    # 
    # See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=ph_learning_rate).minimize(cost)
    
    #Here you check whether the index of the maximum value of the predicted image is equal to the actual labeled image. and both will be a column vector.
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    
    #calculate accuracy across all the given images and average them out. 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return cost, optimizer, accuracy, pred