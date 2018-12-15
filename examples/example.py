from classifyHistology.extract_images import rw_images as extract
from classifyHistology.extract_images import determine_dataset as det

ah={
    'border_step':2,                   # number of pixels to step along tissue border before capturing the next image
    'train_image_size_rc':[100,200],
    'rotate_deg':[-5,0,5],
    'translate_pix_rc':[-5,0,5],
    'reflect_horiz':0,
    'mov_avg_win':100,
    'save_root_dir':'/home/ryan/Documents/Datasets/classify_histology/augmented',
    'image_fill_factor':2/3, #must by <1, >0
    'im_downscale_factor':3,
    'test_dataset_size':0.2, #20% of data will go into test dataset
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
extract.rwImages('/media/ryan/002E-0232/nanozoomer_images/Dataset/Patient101_10-12-2017/Patient101-tumor-1-',ah)

det.determine_dataset('/home/ryan/Documents/Datasets/classify_histology/augmented/dataset_database_info.pkl',augmentation_hyperparameters['test_dataset_size'])

#x_train,y_train=extract.readDataset([32,64,3],'/home/ryan/Documents/Datasets/classify_histology/augmented/train_dataset_database_info.pkl',num_images=1000)

print('test')
