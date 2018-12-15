# This file contains functions to walk along the border (calculated previously)
# of the tissue, take smaller subimages along the border, perform data augmentation,
# then save the augmented image dataset

import itertools
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy as sp
import os
import skimage.transform as trans
from warnings import catch_warnings

def borderWalk(image,im_thresh,row_border,col_border,ah):
    half_bs=math.ceil(ah['border_step']/2)

    # aug_temp_image size needs to be big enough to rotate the train_image by any angle, and shift by the 
    # translation used for data augmentation, and still have the entire train_image fit within the augmentation
    # image
    aug_image_half_width =  int(np.ceil(math.sqrt((ah['train_image_size_rc'][0]/2)**2+(ah['train_image_size_rc'][1]/2)**2)*(1+ah['image_fill_factor']) + np.max([abs(x) for x in ah['translate_pix_rc']]))) # the fill factor term is a bit of a kludge, not exact
    border_index=0
    try:    
        while isSubImageInvalid(border_index,row_border,col_border,aug_image_half_width,image.shape,ah['norm_vec_len_px']):
            border_index = border_index+1
            if border_index>4000:
                raise StopIteration
 
        # first_border_index=border_index
        # while the index falls within the row_border length and while the border position has enough space around it to make a training image
        while not isSubImageInvalid(border_index,row_border,col_border,aug_image_half_width,image.shape,ah['norm_vec_len_px']):
      
            # calculate the orientation of the image, i.e., the angle of the normal vector wrt the image axis
            normal_angle_rad = calcNormalAngle(im_thresh,row_border,col_border,border_index,max([half_bs,10]),ah['norm_vec_len_px'])
      
            # grab the aug_temp_image. It's bigger than needed because we need to rotate
            x_min=col_border[border_index]-aug_image_half_width
            x_max=col_border[border_index]+aug_image_half_width
            y_min=row_border[border_index]-aug_image_half_width
            y_max=row_border[border_index]+aug_image_half_width
            
            aug_temp_image = image[y_min:y_max,x_min:x_max]
            yield aug_temp_image,normal_angle_rad,border_index
            border_index+=ah['border_step']
    except IndexError:
        print('test')

def augmentAndSave(image,normal_angle_rad,fname,dir_name,ah,save_dir='/home/ryan/Documents/Datasets/classify_histology/augmented'):
    
    # get parent name of file
    ind=fname.find('.jpg')
    fname_save=fname[0:ind]
    
    # make a new directory for the augmented data if it doesn't exist
    if not os.path.isdir(dir_name+'/augmented/'):
        os.mkdir(dir_name+'/augmented/')
    
    #sub_image=
    if ah['reflect_horiz']:
        reflect_horiz=[0,1]
    
    # get normal angle in deg
    normal_angle_deg = normal_angle_rad*180/math.pi
    
    # shift the training image down (if background is at top of image) by about 1/6 of its length so that 1/3 of
    # image is background and 2/3 is tissue. Use the normal angle to do this
    adjust=ah['image_fill_factor']-.5
    x_shift=int(np.around(np.real(ah['train_image_size_rc'][1]*adjust*np.exp(normal_angle_rad*1j))))
    y_shift=int(np.around(np.imag(ah['train_image_size_rc'][0]*adjust*np.exp(normal_angle_rad*1j))))
    
    for rot_ang_list in ah['rotate_deg']:
        rot_ang=rot_ang_list-normal_angle_deg
        # rotate image so surface is parallel to columns (plus the data augmentation rotation) and scale
        # so that maximum pixel value is 1.
        rotated = sp.ndimage.rotate(image,-rot_ang)
        
        # we shift the image because we want most of the image to be compose of tissue
        # note this shift assumes that the image has been rotated (ignoring augmentation rotation)
        # by normal_angle_deg, so that the surface of the tissue is parallel to the columns
        col_shift = int(ah['train_image_size_rc'][1]/2*ah['image_fill_factor'])
        
        # calculate the position of the image without translation
        # note that center might change as maximum angle of rotation changes
        center_rc = np.around(rotated.shape[0:2])//2
        row_range=center_rc[0]+np.array([-ah['train_image_size_rc'][0]//2,ah['train_image_size_rc'][0]//2])
        col_range=center_rc[1]-col_shift+np.array([-ah['train_image_size_rc'][1]//2,ah['train_image_size_rc'][1]//2])
        
        aug_rot_ang=int(rot_ang+normal_angle_deg)
        rot_text = num2text(int(rot_ang+normal_angle_deg))       
        translations=itertools.product(ah['translate_pix_rc'],ah['translate_pix_rc'])
        for (dr,dc) in translations:
            this_image = rotated[row_range[0]+dr:row_range[1]+dr,col_range[0]+dc:col_range[1]+dc]
            new_size=[this_image.shape[0]//ah['im_downscale_factor'],this_image.shape[1]//ah['im_downscale_factor'],this_image.shape[2]]
            this_image_resized = np.multiply(trans.resize(this_image, new_size,mode='reflect',anti_aliasing=True),this_image.max()).astype(int) #resize makes it to a float scale 0..1, need to change it back to int 0..255 for saving
            this_image_flipped=np.flipud(this_image_resized)
            prefix='r'+rot_text+'..d'+num2text(dr)+'..d'+num2text(dc)
            augmented=True
            if (aug_rot_ang==0) & (dr == 0) & (dc == 0):
                augmented=False            
            yield this_image_resized,' - '+prefix+'..f0',augmented
            if ah['reflect_horiz']:
                yield this_image_flipped,' - '+prefix+'..f1',True       

def calcNormalAngle(im_thresh,row_border,col_border,border_index,half_bs,norm_vec_len):
    # calculate tangent (using row/col basis)
    trc=[row_border[border_index+half_bs]-row_border[border_index-half_bs],col_border[border_index+half_bs]-col_border[border_index-half_bs]]
    magnitude=math.sqrt(trc[0]**2+trc[1]**2)
    trc=[x/magnitude for x in trc]
    
    # calculate normal (using row/col basis)
    nrc=[trc[1],-trc[0]]
    
    # figure out which way is the tissue, have normal pointing away from the tissue
    nxy=np.array([nrc[1],nrc[0]])
    indexes=np.array(range(0,norm_vec_len,5))
    normal_path_xy=np.around(np.outer(nxy,indexes)).astype(int)
    
    # this expression is asking if the normal we have is pointing into the tissue or outside of the tissue. It's testing if pixels
    # along the normal direction are in the tissue (True value in im_thresh) or background (False value in im_thresh)
    # if the normal is pointing into the tissue, then we switch the direction by 180 degrees because we want the normal pointing outward

    sum1=np.sum(im_thresh[row_border[border_index]+normal_path_xy[1],col_border[border_index]+normal_path_xy[0]])
    sum2=np.sum(im_thresh[row_border[border_index]-normal_path_xy[1],col_border[border_index]-normal_path_xy[0]])
    if sum1>sum2:
        nxy=-nxy
        normal_path_xy=np.around(np.outer(nxy,indexes)).astype(int)
    normal_angle_rad=np.angle(nxy[0]+nxy[1]*1j)
    
    #plt.imshow(im_thresh)
    #plt.gca().invert_yaxis()
    #plt.plot(col_border[border_index]+normal_path_xy[0],row_border[border_index]+normal_path_xy[1])
    #plt.show()
    return normal_angle_rad

# make sure that:
#    1st) the border_index is within the range of the border
#    2nd) any of the future indexing in this code won't go outside the image bounds
#     norm_vec_len term makes sure that the normal vector tip will be inside the image bounds

def isSubImageInvalid(border_index,row_border,col_border,width,shape,norm_vec_len):
    if border_index>=len(row_border):
        return True
    else:
        row=row_border[border_index]
        col=col_border[border_index]
        return (row-width-abs(norm_vec_len)<0) | (row+width+abs(norm_vec_len)>=shape[0]) | (col-width-abs(norm_vec_len)<0) | (col+width+abs(norm_vec_len)>=shape[1])

def num2text(num):
    if num<0:
        sign='-'
    else:
        sign='+'
    mag = str(np.abs(num))
    return sign+mag
