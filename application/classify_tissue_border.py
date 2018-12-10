from classifyHistology.train_net import vars_phs_consts_metrics as vars
from classifyHistology.extract_images import rw_images as extract
import tensorflow as tf

augmentation_hyperparameters={
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
    'norm_vec_len_px':50
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

# load the images to classify
image_location='/media/ryan/002E-0232/nanozoomer_images/testing_application_Patient00-'
normal_angle_rad_list,image_pos_rc_list,images_to_classify=extract.rwImages(image_location,augmentation_hyperparameters,to_mem=True)
images_to_classify=images_to_classify[:,:th['n_input'][0],:th['n_input'][1],:th['n_input'][2]]

# set up the graph, variables, and predictor
tf.reset_default_graph()

# create the variables to load
weights,biases,x,y,ph_is_training = vars.definePhVar(th)

# define performance metrics and optimizer
_, _, _, pred = vars.performanceMetrics(x,y,weights,biases,th,False)

# create the op to save and resotre all the variables
saver = tf.train.Saver()

# model location
model_path='/home/ryan/Dropbox/Code/classifyHistology/TensorBoard/Output04-27PM-December-09-2018/model/model.ckpt'

# launch the model and load the restored variables
with tf.Session() as sess:
    saver.restore(sess, model_path)
    
    prediction = sess.run(pred, feed_dict={x: images_to_classify})
    print(weights)
    print(biases)
print('test')