from classifyHistology.train_net import read_and_reshape_data as read
from classifyHistology.train_net.trainHistNet import train 
from classifyHistology.extract_images import determine_dataset as det
from classifyHistology.extract_images import rw_images as extract


# read the dataframes describing disk locations of the data:
x_train,y_train=extract.readDataset([32,64,3],'/home/ryan/Documents/Datasets/classify_histology/augmented/train_dataset_database_info.pkl',num_images = 1000)
x_test,y_test=extract.readDataset([32,64,3],'/home/ryan/Documents/Datasets/classify_histology/augmented/test_dataset_database_info.pkl')

# training hyperparameteers
th = {
    'training_iters': 200,
    'learning_rate': 0.001,
    'batch_size': 128,
    'n_input': [32,64,3], #is not now used, should be used when defining placeholders
    'n_classes': 2,
    'net':'convNet2',
    'dropout_keep_prob': 0.7}

train(x_train,y_train,x_test,y_test,th)
