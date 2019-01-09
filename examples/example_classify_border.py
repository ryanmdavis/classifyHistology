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
    'border_step':50,                   # number of pixels to step along tissue border before capturing the next image
    'train_image_size_rc':[48,192],
    'rotate_deg':[0],
    'translate_pix_aug_col':[0],
    'translate_pix_aug_row':[0],
    'reflect_horiz':0,
    'mov_avg_win':200,
    'save_root_dir':'/home/ryan/Documents/Datasets/classify_histology/augmented',
    'image_fill_factor':2/3, #must by <1, >0
    'im_downscale_factor':3,
    'test_dataset_size':0.2, #20% of data will go into test dataset
    'norm_vec_len_px':100,
    'threshold_blue':200,
    'strel_size':10
    }

# read the dataframes describing disk locations of the data:
in_image_size=[16,64,3]

# training hyperparameteers
th = {
    'training_iters': 2,
    'learning_rate': 0.001,
    'batch_size': 128,
    'n_input': in_image_size,
    'n_classes': 2,
    'net':'convNet3',
    'dropout_keep_prob': 0.5}

model_path=['/home/ryan/Dropbox/Code/classifyHistology/TensorBoard/Output10-23-23PM-January-02-2019/model/model.ckpt']

x_test,y_test=extract.readDataset([16,64,3],'/home/ryan/Documents/Datasets/classify_histology/augmented3/train_dataset_database_info.pkl', aug = False,str_search='boundry')

probs,is_cancer=ct.classify(model_path,x_test,th)

print('Percent classified as cancer:' + str(sum(is_cancer)/len(is_cancer)))