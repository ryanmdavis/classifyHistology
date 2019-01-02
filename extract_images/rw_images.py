
import os,imageio
import matplotlib.pyplot as plt
from classifyHistology.extract_images import get_tissue_border as border
from classifyHistology.offline_augmentation import augment_and_save as augsave
from classifyHistology.extract_images import progress_bar as prog 
import numpy as np
import pandas as pd
from scipy.misc import imsave
import random as rand
from dask.array.random import choice


# This file does the following:
#    1) Iterate through all of the .png files in the parent folder, going through sub-directories
#    2) Identify the edge of the bladder tissue using morphological operators
#    3) Extract smaller sub-images along the edge of the tissue and save to hard drive
#    Inputs:
#        to_mem - boolean describing if images should be written to hard drive (False) or memory (True)
def rwImages(root_dir,ah,show_steps=False,to_mem=False):
    
    # define the dataframe that will hold all the image labels and locations
    empty_np_array=np.zeros([ah['train_image_size_rc'][0]//ah['im_downscale_factor'],ah['train_image_size_rc'][1]//ah['im_downscale_factor']])
    new_df=pd.DataFrame({'file_loc':['a'],'cancer':[1],'aug':[1],'aug_details':['a'],'patient':['a'],'tissue_loc_id':[1],'section_num':[1]})#,'image':[empty_np_array]})
    data_df=pd.DataFrame()
    #test_df=pd.DataFrame()
    
    # remove the trailing slash if it exists
    if ah['save_root_dir'][-1] is '/':
        ah['save_root_dir']=ah['save_root_dir'][:-1]

    # keep track of image files that were skipped:
    skipped=[]
    
    # for keeping track of tissue sections
    list_of_tissue_sections=[]
    section_num=-1
    
    # each set of augmented (rotated, translated, flipped) gets a unique ID
    tissue_loc_id=-1
    for dir_name, subdirList, fileList in os.walk(root_dir):
        if any(['.jpg' in s for s in fileList]) or any(['.png' in s for s in fileList]):
            # find the name of this subdirectory
            subdir_name=dir_name[dir_name.rfind('/')+1:]
            patient_number=int(subdir_name[subdir_name.find('tient')+5:subdir_name.find('-')])
            
            # give each tissue section a unique identifier
            section_num+=1
            print('section num = ' + str(section_num))
                
            #dir_name='/media/ryan/002E-0232/nanozoomer_images/Patient18/Patient18-tumor1-tr-3-/'
            for fname in fileList:
                # find the file type. We know it has to be a .jpg or .png
                file_type=''
                if '.jpg' in fname:
                    file_type='.jpg'
                else: 
                    file_type='.png'  
                
                if 'annotated' not in fname:
                    f_path=dir_name+'/'+fname
                    print(f_path[f_path.rfind('Patient'):])
                    
                    # read in image
                    im=imageio.imread(dir_name+'/'+fname)
                    if show_steps:
                        plt.subplot(3,1,1)
                        plt.imshow(im)
                    
                    # get a thresholded image and get the tissue border
                    blue_threshold = ah['threshold_blue']
                    im_thresh=(im[:,:,1]<blue_threshold)
                    tissue_border = border.getTissueBorder(im_thresh,ah)
                    
                    # find the center of mass of the image
                    
                    # walk through all of the pixel paths and find the longest one
                    # we're assuming that the longest path is the one lining the tissue edge
                    pixel_path_row=[]
                    pixel_path_col=[]
                    
                    border_count=0
                    path_max_dim=0
                    
                    # while there is still a tissue border pixel near the edge of the image border
                    while border.getEdgePixelLoc(tissue_border)[0]>0:
                        tissue_border_last,new_pixel_path_row,new_pixel_path_col=border.getPixelPath2(tissue_border)
                        tissue_border = tissue_border_last
                        
                        # the right border has the longest diameter
                        path_width = np.max(new_pixel_path_col)-np.min(new_pixel_path_col)
                        path_height = np.max(new_pixel_path_row)-np.min(new_pixel_path_row)
                        new_path_max_dim = np.sqrt(path_width**2+path_height**2)
                        
                        if new_path_max_dim>path_max_dim:
                            pixel_path_row = new_pixel_path_row
                            pixel_path_col = new_pixel_path_col
                            path_max_dim=new_path_max_dim
                                       
                    # show border if desired
                    if show_steps:
                        plt.subplot(3,1,2)
                        plt.imshow(im)
                        plt.plot(pixel_path_col,pixel_path_row)
                    
                    # now smooth the edge, remove the section at the beginning close to the edge
                    row_path_smoothed=np.around(pd.Series(pixel_path_row).rolling(window=ah['mov_avg_win'], win_type='boxcar').mean().tolist()).astype(int)
                    row_path_smoothed=row_path_smoothed[ah['mov_avg_win']-1:-1]
                    
                    col_path_smoothed=np.around(pd.Series(pixel_path_col).rolling(window=ah['mov_avg_win'], win_type='boxcar').mean().tolist()).astype(int)
                    col_path_smoothed=col_path_smoothed[ah['mov_avg_win']-1:-1]
    
                    # show smoothed border if desired
                    if show_steps:
                        plt.subplot(3,1,3)
                        plt.imshow(im)
                        plt.plot(col_path_smoothed,row_path_smoothed)
                        plt.show()
                    
                    # perform offline augmentation - rotate, translate, reflect, save
                    # grab images one by one
                    sub_image_gen=augsave.borderWalk(im,im_thresh,row_path_smoothed,col_path_smoothed,ah)
                    stop_iteration=False
                    retreived_image=False                
                    
                    # decide if this image will be testing or training               
                    if ah['test_dataset_size']==0: # if all data is going into training file
                        is_train_data=True
                    else:
                        is_train_data=bool(np.random.randint(0,int(1/ah['test_dataset_size']-1)))
                    
                    # This loops through all of the aug_temp_images along the surface of the tissue
                    normal_angle_rad_list=[]   
                    image_pos_rc_list=[]
                    aug_images=np.array([])
                    while 1:
                        try:
                            # get the next image for augmentation
                            aug_temp_image,normal_angle_rad,border_index=next(sub_image_gen)
    
                            # Since we just got a new place on the tissue, update the id 
                            tissue_loc_id+=1
                                                            
                            # record that we retreived an aug_image from the image file
                            retreived_image=True
                            aug_gen=augsave.augmentAndSave(aug_temp_image,normal_angle_rad,fname,dir_name,ah)
                            
                            
                            # This loop loops through the generator that goes through each augentation step
                            # i.e. rotation, translation, flipping
                            while 1:
                                try:
                                    # get the next image from data augmentation
                                    aug_image,aug_details,augmented=next(aug_gen)
                                    
                                    # get the next image for augmentation
                                    normal_angle_rad_list.append(normal_angle_rad)
                                    image_pos_rc_list.append((row_path_smoothed[border_index],col_path_smoothed[border_index]))
                                    
                                    # save the list of images into an array
                                    # if its the first image then set equal, if subsequent then append
                                    if not len(aug_images):
                                        aug_images = np.array(aug_image,ndmin=(len(aug_image.shape)+1))
                                    else:
                                        new_image=np.array(aug_image,ndmin=(len(aug_image.shape)+1))
                                        aug_images = np.append(aug_images,new_image,axis=0)
                    
    
                                    # find the name of the directory to save the new image
                                    new_dir_name=ah['save_root_dir']+dir_name[dir_name.rfind('/'):]+'sec'+str(int(section_num))+'/'+fname[:fname.rfind(file_type)]+'/'+'b'+str(border_index)+'/'
                                    if not os.path.exists(new_dir_name):
                                        os.makedirs(new_dir_name)
                                        
                                    # find out the new file name and save
                                    new_file_name=new_dir_name+fname[:fname.rfind(file_type)]+aug_details+'.png'
                                    imsave(new_file_name,aug_image)
                                    
                                    # find the tissue status, a    
                                    #new_df
                                    new_df['file_loc']=new_file_name
                                    new_df['aug']=augmented
                                    new_df['aug_details']=aug_details
                                    new_df['patient']=patient_number
                                    new_df['cancer']=False if 'normal' in dir_name.lower() else True
                                    new_df['tissue_loc_id']=tissue_loc_id
                                    new_df['section_num']=section_num
                                    data_df=data_df.append(new_df)
                                    
                                    # assign to the right file
                                    #if is_train_data:
                                    #    train_df=train_df.append(new_df)
                                    #else:
                                    #    test_df=test_df.append(new_df)
                                    #save into pandas dataframe patient number, aug details, file location, tissue status
    
                                except StopIteration:
                                    break
                        except StopIteration:
                            # Some images just don't work well with the algorithm. If that happens, just
                            # skip the image and notify the user
                            if not retreived_image:
                                skipped.append(dir_name+'/'+fname)
                                print('skipped file: '+skipped[-1])
                            break

    if not to_mem:
        print('pickling')
        data_df_loc=ah['save_root_dir']+'/dataset_database_info.pkl'
        data_df.to_pickle(data_df_loc)
        
        print('Number of training images: ' + str(len(data_df_loc)))
        return data_df_loc
    else: 
        return normal_angle_rad_list,image_pos_rc_list,aug_images,f_path

# take in a dataframe describing the location of each file on hd of augmented dataset and
# return the data in memory
# Optional Flags:
#    tissue_type: 0 for normal, 1 for cancer, 2 for both
def readDataset(out_image_size,df_loc,randomize=True, num_images=-1, str_search='', tissue_type=2, aug=True):
    df=pd.read_pickle(df_loc)

    if not aug: # do we want translated, rotated, etc?
        df=df[df['aug']==False]
    
    if tissue_type<2: # if tissue_type == 2, then do both tissue types
        df=df[df['cancer']==bool(tissue_type)]
   
    if randomize:
        ordered_indicies=[x for x in range(len(df))]
        random_indicies=[]
        while len(ordered_indicies):
            random_indicies.append(ordered_indicies.pop(rand.randint(0,len(ordered_indicies)-1)))
        random_indicies=random_indicies[0:num_images]
    else:
        random_indicies = list(range(len(df)))

    if str_search:
        contains_search_bool_list=df.iloc[random_indicies]['file_loc'].str.contains(str_search).tolist()
        random_indicies_filt=[random_indicies[x] for x in range(len(random_indicies)) if contains_search_bool_list[x]]
        random_indicies=random_indicies_filt
    
    if num_images == -1 or num_images>=len(random_indicies):
        num_images = len(random_indicies)   

    # dim-(number of trainingexamples, row, col, rgb)
    x_train=np.zeros((num_images,out_image_size[0],out_image_size[1],out_image_size[2]))
    
    # dim-(number of training examples, number of classes)
    y_train=np.zeros((num_images,2), dtype=bool)
    
    for image_num in range(0,num_images):
        if not image_num % 500:
            prog.printProgressBar(image_num, num_images, prefix = 'Progress', suffix = '', decimals = 1, length = 100, fill = '█')
        im=imageio.imread(df.iloc[random_indicies[image_num]]['file_loc'])[0:out_image_size[0],0:out_image_size[1],0:out_image_size[2]]
        x_train[image_num,:,:,:]=im.astype(float)/255.
        y_train[image_num,int(df.iloc[random_indicies[image_num]]['cancer'])]=1
    
    return x_train,y_train

#'/home/ryan/Documents/Datasets/classify_histology/augmented/dataset_database_info.pkl'
def showRandomImages(df_path,aug=False,cancer=True,patient='',title=False):
    grid_size=4
    
    data_df = pd.read_pickle(df_path)
    if not aug: # do we want translated, rotated, etc?
        data_df=data_df[data_df['aug']==False]
    data_df=data_df[data_df['cancer']==cancer]
    if patient: # do we want to look at a certain patient?
        data_df=data_df[data_df['patient']==patient]
    
    print('The selected dataset has ' + str(len(data_df)) + ' images.')
    
    rand_ids = []
    normal_id_ls=[x for x in range(len(data_df))]
    for x in range(0,len(data_df)):
        rand_ids.append(normal_id_ls.pop(rand.randint(0,len(normal_id_ls)-1)))
    
    for image_num in range(grid_size**2):
        path=data_df.iloc[rand_ids[image_num]]['file_loc']
        im=imageio.imread(path)
        ax=plt.subplot(grid_size,grid_size,image_num+1)
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(path[path.rfind('atient')-1:path.rfind('/')+2])  
    plt.show() 
