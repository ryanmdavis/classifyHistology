import tensorflow as tf
import datetime
import os

from classifyHistology.train_net import vars_phs_consts_metrics as vars
from classifyHistology.train_net import read_and_reshape_data as read
from classifyHistology.extract_images import rw_images as rw

# this is the master function that takes in training and test data and labels 
def train(train_x,train_y,test_x,test_y,th,model_init_path=[]):
    tf.reset_default_graph() #need this if running train inside loop
    
    # define weights and biases variables based on training hyperparameters (th)
    # also define the placeholders x and y 
    weights,biases,x,y,ph_is_training,ph_learning_rate = vars.definePhVar(th)
    
    # define performance metrics and optimizer
    cost, optimizer, accuracy,pred = vars.performanceMetrics(x,y,weights,biases,th,ph_is_training,ph_learning_rate)
    
    # make an op to save all the learned variabes:
    saver = tf.train.Saver()
    
    # train the net
    with tf.Session() as sess:
        output_dir=os.path.abspath('../../TensorBoard/')+'/Output'+datetime.datetime.now().strftime("%I-%M-%S%p-%B-%d-%Y")
            
        with tf.name_scope('performance'):
            # create loss placeholder and summary
            tf_loss_ph = tf.placeholder(tf.float32, shape=None, name='loss_summary')
            tf_loss_summary=tf.summary.scalar('train_loss',tf_loss_ph)
            
            # create validation loss placeholder and smmary
            tf_valid_loss_ph = tf.placeholder(tf.float32, shape=None, name='valid_loss_summary')
            tf_valid_loss_summary=tf.summary.scalar('test_loss',tf_valid_loss_ph)
            
            # create training accuracy placeholder and summary
            tf_acc_ph = tf.placeholder(tf.float32, shape=None, name='accuracy_summary')
            tf_acc_summary=tf.summary.scalar('train_accuracy',tf_acc_ph)
            
            # create test accuracy placeholder and summary
            tf_test_acc_ph = tf.placeholder(tf.float32, shape=None, name='test_accuracy_summary')
            test_acc_summary=tf.summary.scalar('test_accuracy',tf_test_acc_ph)
    
        # merge summaries together
        performance_summaries = tf.summary.merge([tf_loss_summary,tf_valid_loss_summary,tf_acc_summary,test_acc_summary])
            
        # initialize filewriters
        summary_writer = tf.summary.FileWriter(output_dir, sess.graph)
        
        # Initialize or load the variables
        if model_init_path:
            saver = tf.train.Saver()
            saver.restore(sess, model_init_path)
        else:
            init = tf.global_variables_initializer()
            sess.run(init) 
        for i in range(th['training_iters']):
            #get learning rate here:
            learning_rate_double=getLearningRate(i,th['learning_rate'])
            for batch in range(len(train_x)//th['batch_size']):               
                batch_x = train_x[batch*th['batch_size']:min((batch+1)*th['batch_size'],len(train_x))]
                batch_y = train_y[batch*th['batch_size']:min((batch+1)*th['batch_size'],len(train_y))]    
                # Run optimization op (backprop).
                # Calculate batch loss and accuracy
                opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,ph_is_training:True,ph_learning_rate:learning_rate_double})
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y,ph_is_training:False})
    
            print("Iter " + str(i) + ", Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))
            print("Learning rate = " + str(learning_rate_double))
            print("Optimization Finished!")
    
            # Calculate accuracy for all 10000 mnist test images
            # sess and run evaluate and returns the tensors in the list in the first argument
            test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_x,y : test_y,ph_is_training:False})
            print("Testing Accuracy:","{:.5f}".format(test_acc))
            
            # calculate the summary statistics    
            summ = sess.run(performance_summaries, feed_dict={tf_loss_ph:loss, tf_valid_loss_ph:valid_loss, tf_acc_ph:acc, tf_test_acc_ph:test_acc})
            #summ = sess.run(performance_summaries, feed_dict={tf_loss_ph:loss, tf_acc_ph:acc})
            summary_writer.add_summary(summ,i)
        summary_writer.close()
               
        os.mkdir(output_dir+'/model')
        save_path = saver.save(sess, output_dir+"/model/model.ckpt")
        print('save path: '+ save_path)
        
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: test_x, y: test_y, ph_is_training:False}))
    tensorboard_loc='pipenv run python -m tensorboard.main --logdir='+output_dir
    print(tensorboard_loc)
    return output_dir
#pipenv run python -m tensorboard.main --logdir=drop1:/home/ryan/Dropbox/Code/classifyHistology/TensorBoard/Output04-38-37PM-December-15-2018,drop0.8:/home/ryan/Dropbox/Code/classifyHistology/TensorBoard/Output05-35-24PM-December-15-2018,drop0.6:/home/ryan/Dropbox/Code/classifyHistology/TensorBoard/Output06-34-47PM-December-15-2018,drop0.4:/home/ryan/Dropbox/Code/classifyHistology/TensorBoard/Output07-34-02PM-December-15-2018

def getLearningRate(i,learning_rate):

    if type(learning_rate) is list:
        iter_thresh=[learning_rate[x][0] for x in range(len(learning_rate))]
        learning_rate=[learning_rate[x][1] for x in range(len(learning_rate))]
        iter_index=0
        while not ((i < iter_thresh[iter_index]) or (iter_thresh[iter_index]==-1)):
            iter_index+=1
        return learning_rate[iter_index]
    elif type(learning_rate) is float:
        return learning_rate