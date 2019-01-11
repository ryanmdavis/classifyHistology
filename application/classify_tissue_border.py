from classifyHistology.train_net import vars_phs_consts_metrics as vars
from classifyHistology.train_net import functions as func
from classifyHistology.extract_images import rw_images as extract
from classifyHistology.application import net_plot as netplot
from classifyHistology.application import classify_tissue as ct
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

ah={
    'border_step':20,                   # number of pixels to step along tissue border before capturing the next image
    'train_image_size_rc':[48,192],
    'rotate_deg':[0],
    'translate_pix_aug_col':[0],
    'translate_pix_aug_row':[0],
    'reflect_horiz':0,
    'mov_avg_win':100,
    'save_root_dir':'/home/ryan/Documents/Datasets/classify_histology/augmented3',
    'image_fill_factor':3/4, #must by <1, >0
    'im_downscale_factor':3,
    'test_dataset_size':0.4, #20% of data will go into test dataset
    'norm_vec_len_px':100,
    'threshold_blue':200,
    'strel_size':10
    }

# training hyperparameteers
th = {
    'training_iters': 2,
    'learning_rate': 0.001,
    'batch_size': 128,
    'n_input': [16,64,3],
    'n_classes': 2,
    'net':'convNet3',
    'dropout_keep_prob': 0.5}

dp = {
    'annotation_offset1_px': 30,
    'annotation_offset2_px': 70,
    'mov_med_filt_width': 5}

# load the model path
model_path=['/home/ryan/Dropbox/Code/classifyHistology/TensorBoard/Output11-39-49AM-January-06-2019/model/model.ckpt']
# #model_path=['/home/ryan/Dropbox/Code/classifyHistology/TensorBoard/Output09-43-53PM-December-17-2018/model/model.ckpt','/home/ryan/Dropbox/Code/classifyHistology/TensorBoard/Output12-22-08AM-December-18-2018/model/model.ckpt','/home/ryan/Dropbox/Code/classifyHistology/TensorBoard/Output02-58-28AM-December-18-2018/model/model.ckpt'] #EOD 12/17
# #model_path=['/home/ryan/Dropbox/Code/classifyHistology/TensorBoard/Output10-05-07PM-December-19-2018/model/model.ckpt','/home/ryan/Dropbox/Code/classifyHistology/TensorBoard/Output07-56-55AM-December-20-2018/model/model.ckpt'

# load the images to classify
# image_location='/media/ryan/002E-0232/nanozoomer_images/Application_Data/patient180-tumor1-tr-3-test'
# image_location='/media/ryan/002E-0232/nanozoomer_images/Application_Data/Patient18-normal4-tl-1-'
# image_location='/media/ryan/002E-0232/nanozoomer_images/Application_Data/large_dataset/Patient001'
# image_location='/media/ryan/002E-0232/nanozoomer_images/Application_Data/Patient18-tumor5-br-2-'
# image_location='/media/ryan/002E-0232/nanozoomer_images/Application_Data/Patient18-tumor5-bl-1-'
# image_location='/media/ryan/002E-0232/nanozoomer_images/Application_Data/Patient101-normal-1-' # this is the patient where I get the large dataset from
# image_location='/media/ryan/002E-0232/nanozoomer_images/Application_Data/Patient101-tumor-boundry-1-'
# image_location='/media/ryan/002E-0232/nanozoomer_images/Application_Data/Patient101-tumor-boundry-1-4'

# image_location='/media/ryan/002E-0232/nanozoomer_images/Application_Data/Patient101-tumor-1-'
image_location='/media/ryan/002E-0232/nanozoomer_images/Application_Data/Patient18-normal3-tr-4-'

normal_angle_rad_list,image_pos_rc_list,images_non_standardized,f_path=extract.rwImages(image_location,ah,to_mem=True,show_steps=False)
images_to_classify=ct.standardizeImages(images_non_standardized,ah['save_root_dir'])

probs,is_cancer=ct.classify(model_path,images_to_classify,th)
netplot.displayAnnotated(f_path,normal_angle_rad_list,image_pos_rc_list,probs,f_path,dp)
