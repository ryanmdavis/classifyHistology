from skimage.morphology import erosion,dilation,closing,opening, disk
import matplotlib.pyplot as plt
import numpy as np
def getTissueBorder(bin_img):
    strel5 = disk(5)
    strel1 = disk(1)
    
    # smooth the edge
    opened = opening(bin_img, strel5) #smoothes the edges a little bit so we get nice 1-pixel-wide paths
    closed = closing(opened,strel5)
    
    # erode and calculate the tissue border image
    eroded = erosion(closed, strel1)
    border = closed^eroded
    # show edge extraction:
    # plt.imshow(border[70:220,550:750])
    return border

def getPixelPath(edge_image):
    
    # find an initial pixel to start to calculate a string of pixels
    # But don't let it be on the edge otherwise the 3x3 will reference an invalid pixel
    try:
        initial_pixel=np.array([0,0])
        while (initial_pixel[0]==0) | (initial_pixel[1]==0):
            nonzero_pixels=np.nonzero(edge_image)
            initial_pixel=(nonzero_pixels[0][0],nonzero_pixels[1][0])
            edge_image[initial_pixel]=False
    except IndexError:
        print('test')
                
    
    pixel_locator_col=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    pixel_locator_row=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    
    row=[initial_pixel[0]]
    col=[initial_pixel[1]]
    current_pixel=initial_pixel
    
    
#     image3x3=edge_image[row[-1]-1:row[-1]+2,col[-1]-1:col[-1]+2]
    image3x3, edge_image=getImage3x3(edge_image,row,col)
        
    #now calculate the location of the next pixel in the string        
    while np.sum(image3x3) & ~(row[-1]==0) & ~(col[-1]==0) & (np.sum(image3x3.shape) == 6):
        drow=np.sum(image3x3*pixel_locator_row)
        dcol=np.sum(image3x3*pixel_locator_col)
        row.append(row[-1]+drow)
        col.append(col[-1]+dcol)
        edge_image[row[-1],col[-1]]=False
        image3x3, edge_image=getImage3x3(edge_image,row,col)
        #image3x3=edge_image[row[-1]-1:row[-1]+2,col[-1]-1:col[-1]+2]
    
    return edge_image,row,col

#on rare occasions, image3x3 has more than 1 true, so removed one true in edge image
# and recalculate image 3x3
def getImage3x3(edge_image,row,col):
    image3x3=edge_image[row[-1]-1:row[-1]+2,col[-1]-1:col[-1]+2]
    while image3x3.sum()>1:
        edge_image[row[-1]-1+np.nonzero(image3x3)[0][0],col[-1]-1+np.nonzero(image3x3)[1][0]]=False
        image3x3=edge_image[row[-1]-1:row[-1]+2,col[-1]-1:col[-1]+2]
    return image3x3, edge_image