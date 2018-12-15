from skimage.morphology import erosion,dilation,closing,opening, disk
import matplotlib.pyplot as plt
import numpy as np
def getTissueBorder(bin_img,ah):
    strel5 = disk(ah['strel_size'])
    strel1 = disk(1)
    
    # smooth the edge
    closed = closing(bin_img,strel5)
    opened = opening(closed, strel5) #smoothes the edges a little bit so we get nice 1-pixel-wide paths

    
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
        #if ((row[-1]==616) and (col[-1]==1484)):
        #    print('test')
        image3x3, edge_image=getImage3x3(edge_image,row,col)
        #image3x3=edge_image[row[-1]-1:row[-1]+2,col[-1]-1:col[-1]+2]
    
    return edge_image,row,col

def getPixelPath2(edge_image):
    
    #we want to find an initial pixel that is one pixel removed from the edge,
    # so that we can trance it from edge to edge
 
    # set all edge pixels on the edge of the image as zero, since we need a 3x3
    edge_image=removeBorderPixels(edge_image)

    # get an initial pixel that is one away from the border
    initial_pixel=getEdgePixelLoc(edge_image)
    edge_image[initial_pixel[0],initial_pixel[1]]=False    
    
    edge_image,row,col=tracePath(edge_image,initial_pixel)
    return edge_image,row,col
    
def tracePath(edge_image,initial_pixel):
    pixel_locator_col=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    pixel_locator_row=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    
    row=[initial_pixel[0]]
    col=[initial_pixel[1]]
    current_pixel=initial_pixel
    

    image3x3=edge_image[row[-1]-1:row[-1]+2,col[-1]-1:col[-1]+2]

        

    #now calculate the location of the next pixel in the string        
    while np.sum(image3x3)==1 & ~(row[-1]==0) & ~(col[-1]==0) & (np.sum(image3x3.shape) == 6):
        drow=np.sum(image3x3*pixel_locator_row)
        dcol=np.sum(image3x3*pixel_locator_col)
        row.append(row[-1]+drow)
        col.append(col[-1]+dcol)
        edge_image[row[-1],col[-1]]=False
        #if ((row[-1]==616) and (col[-1]==1484)):
        #    print('test')
        image3x3=edge_image[row[-1]-1:row[-1]+2,col[-1]-1:col[-1]+2]

    # here test if image3x3.sum>2 and if so call tracepath in for loop. keep longest path
    longest_row=[]
    longest_col=[]
    if image3x3.sum()>1:
        nonzero=image3x3.nonzero()
        path_max_dim=0
        # now recursively call tracePath for each possible starting point in image3x3
        for init_row,init_col in zip(nonzero[0],nonzero[1]):
            new_edge_image=edge_image[:]
            initial_pixel[0]=row[-1]+init_row-1
            initial_pixel[1]=col[-1]+init_col-1
            new_edge_image[initial_pixel[0],initial_pixel[1]]=False
            new_edge_image,new_path_row,new_path_col=tracePath(new_edge_image,initial_pixel) #feed nonzeros into initial_pixel
            # for loop here
            # the right border has the longest diameter

            path_width = np.max(new_path_col)-np.min(new_path_col)
            path_height = np.max(new_path_row)-np.min(new_path_row)
            new_path_max_dim = np.sqrt(path_width**2+path_height**2)
            
            if new_path_max_dim>path_max_dim:
                pixel_path_row = new_path_row
                pixel_path_col = new_path_col
                edge_image=new_edge_image
                path_max_dim=new_path_max_dim
                
        row = row+pixel_path_row
        col = col+pixel_path_col    
    return edge_image,row,col

def getPixelPath3(edge_image):
    
    # find an initial pixel to start to calculate a string of pixels
    # But don't let it be on the edge otherwise the 3x3 will reference an invalid pixel
    # Only consider edge_image edges that make it to the edge of the image since
    # those will be the actual edges that we care about

    # set all edge pixels on the edge of the image as zero, since we need a 3x3
    edge_image=removeBorderPixels(edge_image)

    # get an initial pixel that is one away from the border
    initial_pixel=getEdgePixelLoc(edge_image)
    edge_image[initial_pixel[0],initial_pixel[1]]=False            
    
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
        #if ((row[-1]==616) and (col[-1]==1484)):
        #    print('test')
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

def removeBorderPixels(edge_image):
    edge_image[:,0]=False
    edge_image[:,edge_image.shape[1]-1]=False
    edge_image[0,:]=False
    edge_image[edge_image.shape[0]-1,:]=False
    return edge_image

# get a pixel with value true that is one pixel away from the border of edge_image
def getEdgePixelLoc(edge_image):
    if (edge_image[1,:]==True).sum():
        col_nz=edge_image[1,:].nonzero()[0]
        edge_col=col_nz[0]
        edge_row=1
    elif (edge_image[edge_image.shape[0]-2,:]==True).sum():
        col_nz=edge_image[edge_image.shape[0]-2,:].nonzero()[0]
        edge_col=col_nz[0]
        edge_row=edge_image.shape[0]-2
    elif (edge_image[:,1]==True).sum():
        row_nz=edge_image[:,1].nonzero()[0]
        edge_row=row_nz[0]
        edge_col=1
    elif (edge_image[:,edge_image.shape[1]-2]==True).sum():
        row_nz=edge_image[:,edge_image.shape[1]-2].nonzero()[0]
        edge_row=row_nz[0]
        edge_col=edge_image.shape[1]-2
    else:
        edge_row=-1
        edge_col=-1
    
    return [edge_row,edge_col]  