import pandas as pd
import random as rand

# We have about twice as many normal samples as cancer samples, so choose a
# subset of some random normal samples to even things out
def determine_dataset(db_file_loc,percent_testing):
    data_df = pd.read_pickle(db_file_loc)
    
    # determine number of datasets
    num_cancer=len(data_df[(data_df['cancer']==True) & (data_df['aug']==0)])
    num_normal=len(data_df[(data_df['cancer']==False) & (data_df['aug']==0)])
    
    # get set of unique IDs for normal
    normal_id_ls=list(set(data_df[(data_df['cancer']==False)]['tissue_loc_id']))
    
    # get random normal tissue samples
    normal_ids = []
    for x in range(0,num_cancer):
        normal_ids.append(normal_id_ls.pop(rand.randint(0,len(normal_id_ls)-1)))
    
    # now calculate two equal-sized datasets for cancer and normal
    cancer_data_df=data_df[(data_df['cancer']==True)]# & (data_df['tissue_loc_id'].isin(normal_ids))]
    normal_data_df=data_df[(data_df['cancer']==False) & (data_df['tissue_loc_id'].isin(normal_ids))]
    
    # now get ids for equal sized normal and cancer datasets
    normal_id_ls=list(set(cancer_data_df['tissue_loc_id']))
    cancer_id_ls=list(set(normal_data_df['tissue_loc_id']))
    
    # put a random X% of uniue ideas inteo the test set, and others stay in training set
    normal_test_ids=[]
    cancer_test_ids=[]
    for x in range(0,int(len(normal_id_ls)*percent_testing)):
    #for x in range(0,134):
        print(x)
        normal_test_ids.append(normal_id_ls.pop(rand.randint(0,len(normal_id_ls)-1)))
        cancer_test_ids.append(cancer_id_ls.pop(rand.randint(0,len(cancer_id_ls)-1)))
    normal_train_ids=normal_id_ls
    cancer_train_ids=cancer_id_ls
    
    # get testing dataframe
    normal_test_df=data_df[data_df['tissue_loc_id'].isin(normal_test_ids) & (data_df['aug']==False)]
    cancer_test_df=data_df[data_df['tissue_loc_id'].isin(cancer_test_ids) & (data_df['aug']==False)]
    test_df=normal_test_df.append(cancer_test_df)

    # get training dataframe
    normal_train_df=data_df[data_df['tissue_loc_id'].isin(normal_train_ids)]
    cancer_train_df=data_df[data_df['tissue_loc_id'].isin(cancer_train_ids)]
    train_df=normal_train_df.append(cancer_train_df)
    
    # now test and train_df hold paths to files with the testing and training data
    # test_df has non-augmented (rotated, translated,flipped) data
    
    # save it all to the HD
    train_file_path=db_file_loc[:db_file_loc.rfind('/')+1]+'train_'+db_file_loc[db_file_loc.rfind('/')+1:]
    test_file_path=db_file_loc[:db_file_loc.rfind('/')+1]+'test_'+db_file_loc[db_file_loc.rfind('/')+1:]
    test_df.to_pickle(test_file_path)
    train_df.to_pickle(train_file_path)
    
    print('number of training images (normal): ' +str(len(normal_train_df)))
    print('number of training images (cancer): ' +str(len(cancer_train_df)))
    print('number of testing images (normal): ' +str(len(normal_test_df)))
    print('number of testing images (cancer): ' +str(len(cancer_test_df)))
    
    return train_file_path,test_file_path
    
def readDatasetDf():
    test_df=pd.read_pickle('/home/ryan/Documents/Datasets/classify_histology/augmented/test_dataset_database_info.pkl')
    train_df=pd.read_pickle('/home/ryan/Documents/Datasets/classify_histology/augmented/train_dataset_database_info.pkl')
    
    return test_df,train_df
