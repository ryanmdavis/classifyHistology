from classifyHistology.extract_images import rw_images as extract

df_path='/home/ryan/Documents/Datasets/classify_histology/augmented3/train_dataset_database_info.pkl'

extract.showRandomImages(df_path,cancer=False,aug=True)