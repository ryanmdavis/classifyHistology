from classifyHistology.train_net import read_and_reshape_data as read
from classifyHistology.train_net.trainHistNet import train 
from classifyHistology.extract_images import determine_dataset as det
from classifyHistology.extract_images import rw_images as extract

# read the dataframes describing disk locations of the data:
in_image_size=[16,64,3]
x_train,y_train=extract.readDataset('/home/ryan/Documents/Datasets/classify_histology/augmented/train_dataset_database_info.pkl',in_image_size)
x_test,y_test=extract.readDataset('/home/ryan/Documents/Datasets/classify_histology/augmented/test_dataset_database_info.pkl',in_image_size)

# training hyperparameteers
th = {
    'training_iters': 150,
    'learning_rate': [(20,0.001),(150,0.0001),(-1,0.00005)],
    'batch_size': 64,
    'n_input': in_image_size,
    'n_classes': 2,
    'net':'convNet3',
    'dropout_keep_prob': .6}

# tb_loc=train(x_train,y_train,x_test,y_test,th)
with open("/home/ryan/Documents/Datasets/classify_histology/augmented/train_loc.txt","w") as f:
    f.write('pipenv run python -m tensorboard.main --logdir=')

tb_loc_list=[]
for keep_prob in [.6]:
    th['dropout_keep_prob']=keep_prob
    tb_loc=train(x_train,y_train,x_test,y_test,th)
    with open("/home/ryan/Documents/Datasets/classify_histology/augmented/train_loc.txt","a+") as f:
        f.write('drop'+str(int(th['dropout_keep_prob']*10))+':'+tb_loc+',')
