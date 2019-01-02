from classifyHistology.extract_images import rw_images as extract
from classifyHistology.extract_images import determine_dataset as det

ah={
    'border_step':50,                   # number of pixels to step along tissue border before capturing the next image
    'train_image_size_rc':[100,200],
    'rotate_deg':[-5,0,5],
    'translate_pix_aug_col':[-5,0,5],
    'translate_pix_aug_row':[0],
    'reflect_horiz':1,
    'mov_avg_win':100,
    'save_root_dir':'/home/ryan/Documents/Datasets/classify_histology/augmented2',
    'image_fill_factor':2/3, #must by <1, >0
    'im_downscale_factor':3,
    'test_dataset_size':0.4, #20% of data will go into test dataset
    'norm_vec_len_px':50,
    'threshold_blue':200,
    'strel_size':5
    }


# augmentation_hyperparameters={
#     'border_step':50,                   # number of pixels to step along tissue border before capturing the next image
#     'train_image_size_rc':[100,200],
#     'rotate_deg':[0],
#     'translate_pix_rc':[0],
#     'reflect_horiz':1,
#     'mov_avg_win':100,
#     'save_root_dir':'/home/ryan/Documents/Datasets/classify_histology/augmented',
#     'image_fill_factor':2/3, #must by <1, >0
#     'im_downscale_factor':3,
#     'test_dataset_size':0.0, #20% of data will go into test dataset
#     'norm_vec_len_px':50
#     }

# read all of the ?H&E images, walk down the tissue surface, get training images and perform data augmentation
data_df_loc=extract.rwImages('/media/ryan/002E-0232/nanozoomer_images/Dataset',ah)
# extract.rwImages('/media/ryan/002E-0232/nanozoomer_images/Dataset/Patient18/Patient18-normal3-br-2-lowQual',ah)

train_file_path,test_file_path=det.determine_dataset(data_df_loc,ah['test_dataset_size'])


