# For this example, call rw_images with the exact same augmentation hyperparameters but in one case read directly from memory and
# in the other case save to hard drive, then re-read. Use same model to classify both cases. The result should be the same in both
# cases

from classifyHistology.extract_images import rw_images as extract
from classifyHistology.extract_images import determine_dataset as det
from classifyHistology.train_net import read_and_reshape_data as read
from classifyHistology.train_net.trainHistNet import train 
from classifyHistology.application import classify_tissue as ct
import matplotlib.pyplot as plt

ah={
    'border_step':20,                   # number of pixels to step along tissue border before capturing the next image
    'train_image_size_rc':[48,192],
    'rotate_deg':[0],
    'translate_pix_aug_col':[0],
    'translate_pix_aug_row':[0],
    'reflect_horiz':0,
    'mov_avg_win':100,
    'save_root_dir':'/home/ryan/Documents/Datasets/classify_histology/compare/',
    'image_fill_factor':3/4, #must by <1, >0
    'im_downscale_factor':3,
    'test_dataset_size':0.4, #20% of data will go into test dataset
    'norm_vec_len_px':50,
    'threshold_blue':200,
    'strel_size':5
    }
# read the dataframes describing disk locations of the data:
in_image_size=[16,64,3]

# training hyperparameteers
th = {
    'training_iters': 150,
    'learning_rate': [(5,0.001),(50,0.0001),(-1,0.00005)],
    'batch_size': 128,
    'n_input': in_image_size,
    'n_classes': 2,
    'net':'convNet3',
    'dropout_keep_prob': .6}

model_path=['/home/ryan/Dropbox/Code/classifyHistology/TensorBoard/Output10-23-23PM-January-02-2019/model/model.ckpt']
image_location='/media/ryan/002E-0232/nanozoomer_images/Application_Data/Patient101-tumor-boundry-1-4'
#################
# HD
#################
data_df_loc=extract.rwImages(image_location,ah,show_steps=True)
x_test,y_test=extract.readDataset(data_df_loc,randomize=False,standardize=True)
probs_hd,is_cancer_hd=ct.classify(model_path,x_test,th)
    
print('Percent classified as cancer via HD:' + str(sum(is_cancer_hd)/len(is_cancer_hd)))

#################
# Memory
#################
normal_angle_rad_list,image_pos_rc_list,images_non_standardized,f_path=extract.rwImages(image_location,ah,to_mem=True)
images_to_classify=ct.standardizeImages(images_non_standardized,ah['save_root_dir'])
probs_mem,is_cancer_mem=ct.classify(model_path,images_to_classify,th)

print('Percent classified as cancer via memory:' + str(sum(is_cancer_mem)/len(is_cancer_mem)))

for im_num in range(images_to_classify.shape[0]):
    ax1=plt.subplot(images_to_classify.shape[0],2,2*im_num+1)
    plt.imshow(x_test[im_num,:,:,:])
    if im_num==0:
        ax1.set_title("from HD")
    ax2=plt.subplot(images_to_classify.shape[0],2,2*im_num+2)
    plt.imshow(images_to_classify[im_num,:,:])
    if im_num==0:
        ax2.set_title("from Memory")
plt.show()
