from classifyHistology.extract_images import rw_images as extract
from matplotlib import pyplot as plt

x_test,y_test=extract.readDataset([16,64,3],'/home/ryan/Documents/Datasets/classify_histology/augmented3/train_dataset_database_info.pkl', aug = False,str_search='boundry')

grid_size=5
for image_num in range(grid_size**2):
    ax=plt.subplot(grid_size,grid_size,image_num+1)
    ax.imshow(x_test[image_num,:,:,:])
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.show()
print('test')