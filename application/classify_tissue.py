import tensorflow as tf
from classifyHistology.train_net import vars_phs_consts_metrics as vars
from classifyHistology.application import net_plot as netplot
import numpy as np


def classify(model_path,images_to_classify,th):
    # set up the graph, variables, and predictor
    tf.reset_default_graph()
    
    # create the variables to load
    weights,biases,x,y,ph_is_training,ph_learning_rate = vars.definePhVar(th)
    
    # define performance metrics and optimizer
    pred = vars.performanceMetrics(x,y,weights,biases,th,False,ph_learning_rate)[3] #predictor is 4th output in tuple
    
    # create the op to save and resotre all the variables
    #saver = tf.train.Saver({'weigths': weights, 'biases': biases})
    saver = tf.train.Saver()
    
    probs=np.zeros((images_to_classify.shape[0],1))
    # launch the model and load the restored variables
    for model_num in range(len(model_path)):
        with tf.Session() as sess:
            saver.restore(sess, model_path[model_num])
        
            # predict for input image    
            prediction = sess.run(pred, feed_dict={x: images_to_classify, ph_is_training:False})
            probs_to_append=tf.nn.softmax(prediction).eval()
            probs=np.append(probs,probs_to_append[:,1].reshape((-1,1)),axis=1)
    # probs=sig.medfilt(probs[:,1:],kernel_size=[3,1])
    probs=probs[:,1:]
    is_cancer=probs>0.5
    
    return probs,is_cancer