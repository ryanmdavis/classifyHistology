from classifyHistology.extract_images import rw_images as extract

df_path='/home/ryan/Documents/Datasets/classify_histology/augmented/train_dataset_database_info.pkl'

extract.showRandomImages(df_path,cancer=False,patient=101,aug=True)