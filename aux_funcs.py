from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageFilter
from scipy import signal
from scipy import stats
import numpy as np
import scipy as sp
from scipy.ndimage import convolve

class color:
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'


def read_dataset(str, x_len, y_len, patch_sz,show_gray=0):
    """
    Create the input image with a black frame. This is due the convolution layers of the network.
    :param str: the input image path
    :param x_len: width of output image (after rescaling the input image)
    :param y_len: height of output image (after rescaling the input image)
    :param patch_sz: the size of the repeating object
    :param show_gray: flag that is '1' only on the cells images
    """

    print("\n===> " + color.YELLOW + "Reading input image..." + color.END)

    I = Image.open(str)
    patch_sz_hf = np.int32(np.floor(patch_sz/2))

    img = I.resize((y_len, x_len),Image.BILINEAR)
    img = np.float32(np.array(img))

    if len(img.shape) >= 3:
        img = img[0:x_len, 0:y_len, 0:3]
    else:
        img_new = np.zeros((img.shape[0],img.shape[1],3))
        for i in range(0,3):
            img_new[:,:,i] = img[0:x_len, 0:y_len]
        img = img_new

    if show_gray:
        img_orig = img
        img_orig[:,:,0] = img[:,:,2]
        img_orig[:,:,1] = img[:,:,2]
    else:
        img_orig = img

    img_orig = img_orig/255
    img_orig = img_orig+np.random.normal(0,0.01,img_orig.shape)
    img_orig_save = img_orig

    dims = np.shape(img_orig_save)

    img_orig_new = np.zeros((dims[0]+4*patch_sz_hf,dims[1]+4*patch_sz_hf,dims[2]))
    img_orig_new[2*patch_sz_hf:-2*patch_sz_hf,2*patch_sz_hf:-2*patch_sz_hf,:] = img_orig_save

    for c in range(0,3):
        img_orig_new[:,:,c] = img_orig_new[:,:,c]

    img_orig = img_orig_new[patch_sz_hf:-patch_sz_hf,patch_sz_hf:-patch_sz_hf]
    img_orig = np.reshape(img_orig,[1,x_len+patch_sz_hf*2,y_len+patch_sz_hf*2,3])

    print("===> "  + color.YELLOW + "Done preparation" + color.END)

    print(color.BLUE + 80*"-" + color.END)

    return img_orig
