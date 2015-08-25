import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt

red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

BLACK = '\033[30m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
WHITE = '\033[37m'

BRIGHT_BLACK = '\033[30;1m'
BRIGHT_RED = '\033[31;1m'
BRIGHT_GREEN = '\033[32;1m'
BRIGHT_YELLOW = '\033[33;1m'
BRIGHT_BLUE = '\033[34;1m'
BRIGHT_MAGENTA = '\033[35;1m'
BRIGHT_CYAN = '\033[36;1m'
BRIGHT_WHITE = '\033[37;1m'

RESET = '\033[0m'

def polar2cart(r, theta_rad, center):
    x = r  * np.cos(theta_rad) + center[0]
    y = r  * np.sin(theta_rad) + center[1]
    return x, y

def plot_imgs(img_data_lst, color=False, interp='none', max_cols=3):
    cnt=len(img_data_lst)
    r,c,n = cnt,cnt,1
    for idx, img_data in enumerate(img_data_lst):
        if idx % max_cols == 0:
            plt.figure(figsize=(10*r,10*c))
        plt.subplot(r,c,idx+1)
        if color:
            #plt.imshow(img_data[0], interpolation='none', vmax=abs(img_data[0]).max(), vmin=-abs(img_data[0]).max())
            plt.imshow(img_data[0], interpolation=interp)
        else:
            plt.imshow(img_data[0], interpolation=interp, cmap='gray',vmin=0,vmax=255)
        plt.title('%s' % (img_data[1])), plt.xticks([]), plt.yticks([])

def fig2data( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_rgb(), dtype=np.uint8 )
    buf.shape = ( w, h, 3 )
 
    # canvas.tostring_rgb give pixmap in RGB mode. Roll the ALPHA channel to have it in RGBA mode
    #buf = np.roll ( buf, 3, axis = 2 )
    
    #buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    #buf = np.roll ( buf, 3, axis = 2 )
    return buf
        
class Image_List:
    """ Image List
        
        Parses a file spec, possibly sorts them, and reads them into an image list.
        user can then use next, prev methods to iterate through images.
    """
    def __init__(self, image_folder_lst, file_spec, doSort=True, overlay=None, color=False):
        self.image_files = []
        for image_folder in image_folder_lst:
            glob_spec = "%s/%s" % (image_folder, file_spec)
            self.image_files += glob.glob(glob_spec)

        if doSort:
            self.sorted = doSort
            self.image_files = sorted(self.image_files)

        self.images = []
        for image_file in self.image_files:
            img = cv2.imread(image_file, -1)
            if len(img.shape) > 2 and color == False:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if overlay is not None:
                img = cv2.bitwise_and(img, overlay)
            self.images.append(img)

        self.image_idx = 0

    def cnt(self):
        return len(self.images)

    def idx(self):
        return self.image_idx

    def sorted(self):
        return self.sorted

    def next(self):
        self.image_idx += 1
        if self.image_idx < len(self.images):
            return self.images[self.image_idx]
        else:
            self.image_idx = len(self.images) - 1
            return None
            
    def prev(self):
        self.image_idx -= 1
        if self.image_idx >= 0:
            return self.images[self.image_idx]
        else:
            self.image_idx = 0
            return None

    def get(self, idx):
        if idx < len(self.images):
            return self.images[idx]
        else:
            return None

    def images(self):
        return self.images      

class Image_Engine:
    """ Image Engine
    """
    def __init__(self, file_spec, is_glob, color=False):
        self.is_glob = is_glob
        if self.is_glob:
            self.file_spec = file_spec
            self.image_files = glob.glob(self.file_spec)
            self.image_files = sorted(self.image_files)
            print "Image Engine initialized with ", file_spec
            print " Total images: ", len(self.image_files)
        else:
            self.cap = cv2.VideoCapture(file_spec)
            print "Image Engine initialized with ", file_spec

        self.image_idx = 0

    def next(self):
        if self.is_glob:
            if self.image_idx < len( self.image_files ):
                img = cv2.imread(self.image_files[self.image_idx], -1)
                ret = True
                self.image_idx += 1
            else:
                ret, img = False, None
        else:
            if self.cap.isOpened():
                ret, img = self.cap.read()
                if ret:
                    self.image_idx += 1
            else:
                ret, img = False, None
        return ret, img
    
    def idx(self):
        return self.image_idx

    def cleanup(self):
        if not self.is_glob:
            self.cap.release()
