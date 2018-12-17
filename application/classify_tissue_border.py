# 1) return the whole smoothed image border (not just the image locations) and the border index
# corresponding to the image locations
# 
# 2) why is the border only covering a small segment of the surface?

from classifyHistology.train_net import vars_phs_consts_metrics as vars
from classifyHistology.train_net import functions as func
from classifyHistology.extract_images import rw_images as extract
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio
import math,cmath,sys

ah={
    'border_step':50,                   # number of pixels to step along tissue border before capturing the next image
    'train_image_size_rc':[100,200],
    'rotate_deg':[0],
    'translate_pix_rc':[0],
    'reflect_horiz':0,
    'mov_avg_win':100,
    'save_root_dir':'/home/ryan/Documents/Datasets/classify_histology/augmented',
    'image_fill_factor':2/3, #must by <1, >0
    'im_downscale_factor':3,
    'test_dataset_size':0.2, #20% of data will go into test dataset
    'norm_vec_len_px':50,
    'threshold_blue':200,
    'strel_size':10
    }

# training hyperparameteers
th = {
    'training_iters': 2,
    'learning_rate': 0.001,
    'batch_size': 128,
    'n_input': [32,64,3],
    'n_classes': 2,
    'net':'convNet2',
    'dropout_keep_prob': 0.5}

#x_test,y_test=extract.readDataset([32,64,3],'/home/ryan/Documents/Datasets/classify_histology/augmented/test_dataset_database_info.pkl')

# load the images to classify
#image_location='/media/ryan/002E-0232/nanozoomer_images/Application_Data/patient180-tumor1-tr-3-test'
#image_location='/media/ryan/002E-0232/nanozoomer_images/Application_Data/Patient18-normal4-tl-1-'
#image_location='/media/ryan/002E-0232/nanozoomer_images/Application_Data/large_dataset/Patient001'
image_location='/media/ryan/002E-0232/nanozoomer_images/Application_Data/Patient18-tumor5-br-2-'
#image_location='/media/ryan/002E-0232/nanozoomer_images/Application_Data/Patient18-tumor5-bl-1-'

#read a raw image. This one is in the testing dataset and is correctly classified with the below predictor:
# im_big=imageio.imread('/media/ryan/002E-0232/nanozoomer_images/Application_Data/Patient18-tumor5-bl-1-3-b931/3 - r+0..d+0..d+0..f0.png')
# im=im_big[0:32,0:64,:].reshape((1,32,64,3))

normal_angle_rad_list,image_pos_rc_list,images_to_classify,f_path=extract.rwImages(image_location,ah,to_mem=True)
images_to_classify=images_to_classify[:,:th['n_input'][0],:th['n_input'][1],:th['n_input'][2]]/255.


# set up the graph, variables, and predictor
tf.reset_default_graph()

# create the variables to load
weights,biases,x,y,ph_is_training = vars.definePhVar(th)

# define performance metrics and optimizer
pred = vars.performanceMetrics(x,y,weights,biases,th,False)[3] #predictor is 4th output in tuple

# create the op to save and resotre all the variables
#saver = tf.train.Saver({'weigths': weights, 'biases': biases})
saver = tf.train.Saver()

# model location
model_path='/home/ryan/Dropbox/Code/classifyHistology/TensorBoard/Output06-48-00PM-December-16-2018/model/model.ckpt'

# launch the model and load the restored variables
with tf.Session() as sess:
    saver.restore(sess, model_path)

    # predict for input image    
    prediction = sess.run(pred, feed_dict={x: images_to_classify, ph_is_training:False})
    probs=tf.nn.softmax(prediction).eval()
    print(probs)
    
#     # predict for test data
#     # Test model
#     correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#     # Calculate accuracy
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#     print("Test Data Accuracy:", accuracy.eval({x: x_test, y: y_test, ph_is_training:False}))
#     prediction = sess.run(pred, feed_dict={x: x_test, ph_is_training:False})
#     probs=tf.nn.softmax(prediction).eval()
#     print(probs)

# read and plot image
im_in_loc=f_path
im=imageio.imread(im_in_loc)
plt.imshow(im)

# find the path corresponding to the surface of the image
imag_vec_rc=[(cmath.exp(1j*normal_angle_rad_list[x]).imag,cmath.exp(1j*normal_angle_rad_list[x]).real) for x in range(len(normal_angle_rad_list))]
annotation_pos_row=[int(imag_vec_rc[x][0]*ah['norm_vec_len_px']+image_pos_rc_list[x][0]) for x in range(len(imag_vec_rc))]
annotation_pos_col=[int(imag_vec_rc[x][1]*ah['norm_vec_len_px']+image_pos_rc_list[x][1]) for x in range(len(imag_vec_rc))]
image_surface_row=[image_pos_rc_list[x][0] for x in range(len(image_pos_rc_list))]
image_surface_col=[image_pos_rc_list[x][1] for x in range(len(image_pos_rc_list))]
plt.subplot('211')
plt.imshow(im)
plt.plot(annotation_pos_col,annotation_pos_row)
plt.plot(image_surface_col,image_surface_row)
plt.subplot('212')
plt.plot(probs[:,0])

# find positions of line annotations
write_loc=f_path[0:f_path.rfind('/')+1]+'annotated-'+f_path[f_path.rfind('/')+1:]
plt.savefig(write_loc)
