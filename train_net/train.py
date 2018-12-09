import tensorflow as tf
import datetime

from classifyHistology.train_net import vars_phs_consts_metrics as vars
from classifyHistology.train_net import read_and_reshape_data as read

# this is the master function that takes in training and test data and labels 
def train(train_X,test_X, train_y,test_y,th):    
    # define weights and biases variables based on training hyperparameters (th)
    # also define the placeholders x and y 
    weights,biases,x,y,ph_is_training = vars.definePhVar(th)
    
    # define performance metrics and optimizer
    
    cost, optimizer, accuracy = vars.performanceMetrics(x,y,weights,biases,th,ph_is_training)
    
    # train the net
    with tf.Session() as sess:
        output_dir='./Output'+datetime.datetime.now().strftime("%I-%M%p-%B-%d-%Y")
            
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
        
        # Initialize the variables
        init = tf.global_variables_initializer()
        
        sess.run(init) 
        for i in range(th['training_iters']):
            for batch in range(len(train_X)//th['batch_size']):
                #print(str(i)+','+str(batch))
                batch_x = train_X[batch*th['batch_size']:min((batch+1)*th['batch_size'],len(train_X))]
                batch_y = train_y[batch*th['batch_size']:min((batch+1)*th['batch_size'],len(train_y))]    
                # Run optimization op (backprop).
                # Calculate batch loss and accuracy
                opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,ph_is_training:True})
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y,ph_is_training:False})
    
            print("Iter " + str(i) + ", Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))
            print("Optimization Finished!")
    
            # Calculate accuracy for all 10000 mnist test images
            # sess and run evaluate and returns the tensors in the list in the first argument
            test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y,ph_is_training:False})
            print("Testing Accuracy:","{:.5f}".format(test_acc))
            
            # calculate the summary statistics    
            summ = sess.run(performance_summaries, feed_dict={tf_loss_ph:loss, tf_valid_loss_ph:valid_loss, tf_acc_ph:acc, tf_test_acc_ph:test_acc})
            summary_writer.add_summary(summ,i)
        summary_writer.close()
    
    print('pipenv run python -m tensorboard.main --logdir=/home/ryan/Dropbox/Code/TF_examples/datacamp_tensorflow_tutorial'+output_dir[1:])
