from classifyHistology.train_net import read_and_reshape_data as read
from classifyHistology.train_net.train import train 

# read in data
train_X,test_X, train_y,test_y = read.readReshapeData('/home/ryan/Downloads/datasets/fashion-mnist-master/data/fashion')

# training hyperparameteers
th = {
    'training_iters': 1,
    'learning_rate': 0.001,
    'batch_size': 128,
    'n_input': [28,28], #is not now used, should be used when defining placeholders
    'n_classes': 10,
    'net':'convNet2',
    'dropout_keep_prob': 0.5}

# train
train(train_X,test_X, train_y,test_y,th)

# (33, 66, 3)