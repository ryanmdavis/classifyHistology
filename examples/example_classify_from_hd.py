from classifyHistology.extract_images import rw_images as extract
from classifyHistology.application import classify_tissue as ct

# training hyperparameters used to train the model
th = {
    'training_iters': 150,
    'learning_rate': [(5,0.001),(50,0.0001),(-1,0.00005)],
    'batch_size': 128,
    'n_input': [16,64,3],
    'n_classes': 2,
    'net':'convNet3',
    'dropout_keep_prob': .6}

dataset_path='/home/ryan/Documents/Datasets/classify_histology/augmented3/dataset_database_info.pkl'
model_path=['/home/ryan/Dropbox/Code/classifyHistology/TensorBoard/Output10-23-23PM-January-02-2019/model/model.ckpt']

[x_test,y_test]=extract.readDataset(th['n_input'],dataset_path, aug = False,str_search='boundry-1-sec19',patient=101,tissue_type=1)


probs,is_cancer=ct.classify(model_path,x_test,th)
correct_prediction = is_cancer.reshape((-1,)) == y_test[:,1].transpose()

print("prediction accuracy = "+str(sum(correct_prediction)/len(correct_prediction)))
print("number of images classified = " + str(len(correct_prediction)))