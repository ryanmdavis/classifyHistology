#import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import imageio
import math,cmath,sys
import matplotlib.cm as cm


# This function was taken directly from:
# https://stackoverflow.com/questions/36505587/color-line-by-third-variable-python
def plotColorline(x,y,c,cmap):
    ca=np.array(c)
   
    c_rgba=cmap((ca*255))
    #c_rgba = cmap((ca-ca.min())/(ca.max()-ca.min()))
    ax = plt.gca()
    for i in np.arange(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], c=c_rgba[i])
    return

def displayAnnotated(image_location,normal_angle_rad_list,image_pos_rc_list,probs,is_cancer,f_path,ah):
    # read and plot image

    im=imageio.imread(image_location)
    plt.imshow(im)
    
    # find the path corresponding to the surface of the image
    imag_vec_rc=[(cmath.exp(1j*normal_angle_rad_list[x]).imag,cmath.exp(1j*normal_angle_rad_list[x]).real) for x in range(len(normal_angle_rad_list))]
    annotation_pos_row=[int(imag_vec_rc[x][0]*ah['norm_vec_len_px']+image_pos_rc_list[x][0]) for x in range(len(imag_vec_rc))]
    annotation_pos_col=[int(imag_vec_rc[x][1]*ah['norm_vec_len_px']+image_pos_rc_list[x][1]) for x in range(len(imag_vec_rc))]
    is_cancer_pos_row=[int(imag_vec_rc[x][0]*ah['norm_vec_len_px']*2+image_pos_rc_list[x][0]) for x in range(len(imag_vec_rc))]
    is_cancer_pos_col=[int(imag_vec_rc[x][1]*ah['norm_vec_len_px']*2+image_pos_rc_list[x][1]) for x in range(len(imag_vec_rc))]
    image_surface_row=[image_pos_rc_list[x][0] for x in range(len(image_pos_rc_list))]
    image_surface_col=[image_pos_rc_list[x][1] for x in range(len(image_pos_rc_list))]
    plt.subplot('211')
    plt.imshow(im)
    #plt.plot(annotation_pos_col,annotation_pos_row)
    #plt.plot(image_surface_col,image_surface_row)
    plotColorline(annotation_pos_col,annotation_pos_row,(255*probs.transpose()[0]).astype(int).tolist(),cm.jet)
    plotColorline(is_cancer_pos_col,is_cancer_pos_row,(255*is_cancer.transpose()[0]).astype(int).tolist(),cm.brg)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    
    plt.subplot('212')
    plt.plot(probs)
    plt.show()
    
    # find positions of line annotations
    write_loc=f_path[0:image_location.rfind('/')+1]+'annotated-'+image_location[image_location.rfind('/')+1:]
    plt.savefig(write_loc)
    
    print("Image written to: "+write_loc)
