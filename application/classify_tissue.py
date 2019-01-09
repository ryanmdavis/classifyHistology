import tensorflow as tf
from classifyHistology.train_net import vars_phs_consts_metrics as vars
from classifyHistology.application import net_plot as netplot
import numpy as np
import os

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
    cancer_probs=probs[:,1]
    is_cancer=cancer_probs>0.5
    
    return cancer_probs,is_cancer

def standardizeImages(images,save_root_dir):
    # moments file must be written for this function to work
    assert os.path.isfile(save_root_dir+'/moments.npy')
    moments=np.load(save_root_dir+'/moments.npy')
    for image_num in range(images.shape[0]):
        im_temp=np.divide(np.subtract(images[image_num,:,:,:],moments[0,:,:,:]),moments[1,:,:,:])
        images[image_num,:,:,:] = im_temp[:]
    return images