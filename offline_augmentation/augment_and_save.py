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
    max_aug_translate=np.absolute(np.array(ah['translate_pix_aug_col'])+np.array(ah['translate_pix_aug_row'])).max()
    aug_image_half_width =  int(np.ceil(math.sqrt((ah['train_image_size_rc'][0]/2)**2+(ah['train_image_size_rc'][1]/2)**2)*(1+ah['image_fill_factor']) + max_aug_translate)) # the fill factor term is a bit of a kludge, not exact
    border_index=0
    try:    
        #while isSubImageInvalid(border_index,row_border,col_border,aug_image_half_width,image.shape,ah['norm_vec_len_px']):
        normal_angle_rad = calcNormalAngle(im_thresh,row_border,col_border,border_index,max([half_bs,10]),ah['norm_vec_len_px'])
        aug_temp_image=isSubImageValid2(border_index,row_border,col_border,image,normal_angle_rad,ah)
        while not aug_temp_image.any():
            border_index = border_index+1
            normal_angle_rad = calcNormalAngle(im_thresh,row_border,col_border,border_index,max([half_bs,10]),ah['norm_vec_len_px'])
            aug_temp_image=isSubImageValid2(border_index,row_border,col_border,image,normal_angle_rad,ah)
            if border_index>4000:
                raise StopIteration
 
        # first_border_index=border_index
        # while the index falls within the row_border length and while the border position has enough space around it to make a training image
        #while not isSubImageInvalid(border_index,row_border,col_border,aug_image_half_width,image.shape,ah['norm_vec_len_px']):
        while aug_temp_image.any():
      
            # calculate the orientation of the image, i.e., the angle of the normal vector wrt the image axis
            #normal_angle_rad = calcNormalAngle(im_thresh,row_border,col_border,border_index,max([half_bs,10]),ah['norm_vec_len_px'])
      
            # grab the aug_temp_image. It's bigger than needed because we need to rotate
            #x_min=col_border[border_index]-aug_image_half_width
            #x_max=col_border[border_index]+aug_image_half_width
            #y_min=row_border[border_index]-aug_image_half_width
            #y_max=row_border[border_index]+aug_image_half_width
            
            #aug_temp_image = image[y_min:y_max,x_min:x_max]
            #aug_temp_image=isSubImageValid2(border_index,row_border,col_border,image,ah)
            yield aug_temp_image,normal_angle_rad,border_index
           
            # Now update the border index, calculate the orientation of the image, i.e., the angle of the normal vector wrt the image axis
            border_index+=ah['border_step']
            normal_angle_rad = calcNormalAngle(im_thresh,row_border,col_border,border_index,max([half_bs,10]),ah['norm_vec_len_px'])
            aug_temp_image=isSubImageValid2(border_index,row_border,col_border,image,normal_angle_rad,ah)
      
    except IndexError:
        print('test')

def augmentAndSave(image,normal_angle_rad,fname,dir_name,ah,save_dir='/home/ryan/Documents/Datasets/classify_histology/augmented'):
    
    # get parent name of file
    ind=fname.find('.jpg')
    fname_save=fname[0:ind]
    
    # make a new directory for the augmented data if it doesn't exist
    #if not os.path.isdir(dir_name+'/augmented/'):
    #    os.mkdir(dir_name+'/augmented/')
    
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
        
        # remove any pixels with value over 255
        rotated[rotated>255]=255
        rotated[rotated<0]=0
        
        # we shift the image because we want most of the image to be compose of tissue
        # note this shift assumes that the image has been rotated (ignoring augmentation rotation)
        # by normal_angle_deg, so that the surface of the tissue is parallel to the columns
        col_shift = int(ah['train_image_size_rc'][1]*(ah['image_fill_factor']-0.5))
        #col_shift = int(ah['train_image_size_rc'][1]/2*ah['image_fill_factor'])
        
        # calculate the position of the image without translation
        # note that center might change as maximum angle of rotation changes
        center_rc = np.around(rotated.shape[0:2])//2
        row_range=center_rc[0]+np.array([-ah['train_image_size_rc'][0]//2,ah['train_image_size_rc'][0]//2])
        col_range=center_rc[1]-col_shift+np.array([-ah['train_image_size_rc'][1]//2,ah['train_image_size_rc'][1]//2])
        
        aug_rot_ang=int(rot_ang+normal_angle_deg)
        rot_text = num2text(int(rot_ang+normal_angle_deg))       
        translations=itertools.product(ah['translate_pix_aug_row'],ah['translate_pix_aug_col'])
        for (dr,dc) in translations:
            this_image = rotated[row_range[0]+dr:row_range[1]+dr,col_range[0]+dc:col_range[1]+dc]
            new_size=[this_image.shape[0]//ah['im_downscale_factor'],this_image.shape[1]//ah['im_downscale_factor'],this_image.shape[2]]
            try:
                this_image_resized = np.multiply(trans.resize(this_image.astype(float)/255., new_size,mode='reflect',anti_aliasing=True),this_image.max()).astype(int) #resize makes it to a float scale 0..1, need to change it back to int 0..255 for saving
            except:
                print('test')
                
            this_image_flipped = np.flipud(this_image_resized)
            prefix='r'+rot_text+'..d'+num2text(dr)+'..d'+num2text(dc)
            augmented=True
            if (aug_rot_ang==0) & (dr == 0) & (dc == 0):
                augmented=False            
            yield this_image_resized,' - '+prefix+'..f0',augmented
            if ah['reflect_horiz']:
                yield this_image_flipped,' - '+prefix+'..f1',True       

def calcNormalAngle(im_thresh,row_border,col_border,border_index,half_bs,norm_vec_len):
    
    #make sure that the border indices are within the range of the border vector
    lower_border_index=max(0,border_index-half_bs)
    upper_border_index=min(border_index+half_bs,len(row_border))
    # calculate tangent (using row/col basis)
    trc=[row_border[upper_border_index]-row_border[lower_border_index],col_border[upper_border_index]-col_border[lower_border_index]]
    #trc=[row_border[border_index+half_bs]-row_border[border_index-half_bs],col_border[border_index+half_bs]-col_border[border_index-half_bs]]
    
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

# Problem Geometry: dN correspond to delta row,col from the border pixel row,col to the
# training image row,col. d1 is top right, d2 is bottom right, d3 is bottom left, d4 is top left
def isSubImageValid2(border_index,row_border,col_border,image,normal_angle,ah):
    
    if border_index>=len(row_border):
        return True
    else:
        # imaginary dimensions correspond to y and row dimensions. Note that here the origin
        # is the border pixel, around which we rotate, so it doesn't show up here.
        d1=(+ah['train_image_size_rc'][1]/2-ah['train_image_size_rc'][1]*(ah['image_fill_factor']-0.5))+1j*(-ah['train_image_size_rc'][0]/2)
        d2=(+ah['train_image_size_rc'][1]/2-ah['train_image_size_rc'][1]*(ah['image_fill_factor']-0.5))-1j*(-ah['train_image_size_rc'][0]/2)
        d3=(-ah['train_image_size_rc'][1]/2-ah['train_image_size_rc'][1]*(ah['image_fill_factor']-0.5))-1j*(-ah['train_image_size_rc'][0]/2)
        d4=(-ah['train_image_size_rc'][1]/2-ah['train_image_size_rc'][1]*(ah['image_fill_factor']-0.5))+1j*(-ah['train_image_size_rc'][0]/2)
        
        min_rdx=min_rdy=10000
        max_rdx=max_rdy=-10000
        for aug_angle in ah['rotate_deg']:
            # calculate the training image corners after the training image has been rotated to the normal direction
            rd1=d1*np.exp(1j*(normal_angle+aug_angle/180.*math.pi))
            rd2=d2*np.exp(1j*(normal_angle+aug_angle/180.*math.pi))
            rd3=d3*np.exp(1j*(normal_angle+aug_angle/180.*math.pi))
            rd4=d4*np.exp(1j*(normal_angle+aug_angle/180.*math.pi))
            
            # Now calculate the maximum possible extent of the image
            # Add in:
            #    1) The rotation buffer because during rotation zero padding will cause
            #     an aliasing artifact. Rotation_buffer makes the image a bit wider to avoid
            #     a useful pixel adjacent to a zeroed pixel
            #    2) Also add any translations for data augmentation
            rotation_buffer=2
            temp_min_rdx=math.floor((np.array([rd1.real,rd2.real,rd3.real,rd4.real]).min()))-rotation_buffer+np.array(ah['translate_pix_aug_col']).min()
            temp_max_rdx=math.ceil((np.array([rd1.real,rd2.real,rd3.real,rd4.real]).max()))+rotation_buffer+np.array(ah['translate_pix_aug_col']).max()
            temp_min_rdy=math.floor((np.array([rd1.imag,rd2.imag,rd3.imag,rd4.imag]).min()))-rotation_buffer+np.array(ah['translate_pix_aug_row']).min()
            temp_max_rdy=math.ceil((np.array([rd1.imag,rd2.imag,rd3.imag,rd4.imag]).max()))+rotation_buffer+np.array(ah['translate_pix_aug_row']).min()
            
            # now if the new min is less than the old min, replace
            min_rdx=temp_min_rdx if temp_min_rdx<min_rdx else min_rdx
            min_rdy=temp_min_rdy if temp_min_rdy<min_rdy else min_rdy
            max_rdx=temp_max_rdx if temp_max_rdx>max_rdx else max_rdx
            max_rdy=temp_max_rdy if temp_max_rdy>max_rdy else max_rdy
                    
        # now show the maximum dimensions of the image
        min_x=min_rdx+col_border[border_index]
        max_x=max_rdx+col_border[border_index]
        min_y=min_rdy+row_border[border_index]
        max_y=max_rdy+row_border[border_index]
    
        #print("min_x: "+str(min_x)+",  max_x: "+str(max_x)+",  min_y: "+str(min_y)+",  max_y: "+str(max_y))
        if min_x < 0 or min_y < 0 or max_x >= image.shape[1] or max_y >= image.shape[0]:
            return np.array([])
        else: # the image is valid, so return it
            # get the 4 corners of the agumented image. Needs to be 
            max_abs_rdy=max([abs(min_rdy),abs(max_rdy)])
            max_abs_rdx=max([abs(min_rdx),abs(max_rdx)])
            aug_temp_image=np.zeros((2*max_abs_rdy+1,2*max_abs_rdx+1,3),dtype=int)
            
            # aug_temp_image[max_abs_rdy+1, max_abs_rdx+1] is the center of the image:
            aug_temp_image[max_abs_rdy+1+min_rdy:max_abs_rdy+1+max_rdy,max_abs_rdx+1+min_rdx:max_abs_rdx+1+max_rdx,:]=image[row_border[border_index]+min_rdy:row_border[border_index]+max_rdy,col_border[border_index]+min_rdx:col_border[border_index]+max_rdx,:]
            return aug_temp_image#.astype(float)#/255.
    print('test')

def num2text(num):
    if num<0:
        sign='-'
    else:
        sign='+'
    mag = str(np.abs(num))
    return sign+mag
