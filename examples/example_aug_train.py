from classifyHistology.extract_images import rw_images as extract
from classifyHistology.extract_images import determine_dataset as det
from classifyHistology.train_net import read_and_reshape_data as read
from classifyHistology.train_net.trainHistNet import train 


ah={
    'border_step':10,                   # number of pixels to step along tissue border before capturing the next image
    'train_image_size_rc':[100,200],
    'rotate_deg':[-10,-5,0,5,10],
    'translate_pix_rc':[0],
    'reflect_horiz':1,
    'mov_avg_win':100,
    'save_root_dir':'/home/ryan/Documents/Datasets/classify_histology/augmented',
    'image_fill_factor':2/3, #must by <1, >0
    'im_downscale_factor':3,
    'test_dataset_size':0.4, #20% of data will go into test dataset
    'norm_vec_len_px':50,
    'threshold_blue':200,
    'strel_size':5
    }

# read all of the ?H&E images, walk down the tissue surface, get training images and perform data augmentation
data_df_loc=extract.rwImages('/media/ryan/002E-0232/nanozoomer_images/Dataset',ah)
# extract.rwImages('/media/ryan/002E-0232/nanozoomer_images/Dataset/Patient18/Patient18-normal3-br-2-lowQual',ah)

train_file_path,test_file_path=det.determine_dataset(data_df_loc,ah['test_dataset_size'])

# read the dataframes describing disk locations of the data:
x_train,y_train=extract.readDataset([32,64,3],'/home/ryan/Documents/Datasets/classify_histology/augmented/train_dataset_database_info.pkl')
x_test,y_test=extract.readDataset([32,64,3],'/home/ryan/Documents/Datasets/classify_histology/augmented/test_dataset_database_info.pkl')

# training hyperparameteers
th = {
    'training_iters': 20,
    'learning_rate': 0.001,
    'batch_size': 128,
    'n_input': [32,64,3], #is not now used, should be used when defining placeholders
    'n_classes': 2,
    'net':'convNet2',
    'dropout_keep_prob': .6}

# tb_loc=train(x_train,y_train,x_test,y_test,th)
with open("/home/ryan/Documents/Datasets/classify_histology/augmented/train_loc.txt","w") as f:
    f.write('pipenv run python -m tensorboard.main --logdir=')

tb_loc_list=[]
for keep_prob in [1,.6,.4]:
    th['dropout_keep_prob']=keep_prob
    tb_loc=train(x_train,y_train,x_test,y_test,th)
    with open("/home/ryan/Documents/Datasets/classify_histology/augmented/train_loc.txt","a+") as f:
        f.write('drop'+str(int(th['dropout_keep_prob']*10))+':'+tb_loc+',')
