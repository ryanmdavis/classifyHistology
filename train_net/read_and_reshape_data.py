from tensorflow.examples.tutorials.mnist import input_data

def readReshapeData(path):
    # extract data
    data = input_data.read_data_sets(path,one_hot=True)
    
    # print info about the datasets
    # Shapes of training set
    print("Training set (images) shape: {shape}".format(shape=data.train.images.shape))
    print("Training set (labels) shape: {shape}".format(shape=data.train.labels.shape))
    
    # Shapes of test set
    print("Test set (images) shape: {shape}".format(shape=data.test.images.shape))
    print("Test set (labels) shape: {shape}".format(shape=data.test.labels.shape))
    
    # Reshape training and testing image
    train_X = data.train.images.reshape(-1, 28, 28, 1)
    test_X = data.test.images.reshape(-1,28,28,1)
    
    # set the correct classes
    train_y = data.train.labels
    test_y = data.test.labels
    
    return train_X,test_X, train_y,test_y