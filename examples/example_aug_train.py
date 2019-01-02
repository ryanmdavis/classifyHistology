from classifyHistology.extract_images import rw_images as extract
from classifyHistology.extract_images import determine_dataset as det
from classifyHistology.train_net import read_and_reshape_data as read
from classifyHistology.train_net.trainHistNet import train 


ah={
    'border_step':15,                   # number of pixels to step along tissue border before capturing the next image
    'train_image_size_rc':[48,192],
    'rotate_deg':[0],
    'translate_pix_aug_col':[-20,-10,0,10],
    'translate_pix_aug_row':[0],
    'reflect_horiz':0,
    'mov_avg_win':50,
    'save_root_dir':'/home/ryan/Documents/Datasets/classify_histology/augmented3',
    'image_fill_factor':3/4, #must by <1, >0
    'im_downscale_factor':3,
    'test_dataset_size':0.4, #20% of data will go into test dataset
    'norm_vec_len_px':50,
    'threshold_blue':200,
    'strel_size':5
    }
# read the dataframes describing disk locations of the data:
in_image_size=[16,64,3]

# read all of the ?H&E images, walk down the tissue surface, get training images and perform data augmentation
data_df_loc=extract.rwImages('/media/ryan/002E-0232/nanozoomer_images/Dataset',ah)
# extract.rwImages('/media/ryan/002E-0232/nanozoomer_images/Dataset/Patient18/Patient18-normal3-br-2-lowQual',ah)

train_file_path,test_file_path=det.determine_dataset(data_df_loc,ah['test_dataset_size'])

x_train,y_train=extract.readDataset(in_image_size,train_file_path)
x_test,y_test=extract.readDataset(in_image_size,test_file_path)

# training hyperparameteers
th = {
    'training_iters': 150,
    'learning_rate': [(5,0.001),(50,0.0001),(-1,0.00005)],
    'batch_size': 128,
    'n_input': in_image_size,
    'n_classes': 2,
    'net':'convNet3',
    'dropout_keep_prob': .6}

# tb_loc=train(x_train,y_train,x_test,y_test,th)
with open("/home/ryan/Documents/Datasets/classify_histology/augmented/train_loc.txt","w") as f:
    f.write('pipenv run python -m tensorboard.main --logdir=')

tb_loc_list=[]
for keep_prob in [.4,.6,1]:
    th['dropout_keep_prob']=keep_prob
    tb_loc=train(x_train,y_train,x_test,y_test,th)
    with open("/home/ryan/Documents/Datasets/classify_histology/augmented/train_loc.txt","a+") as f:
        f.write('drop'+str(int(th['dropout_keep_prob']*10))+':'+tb_loc+',')
