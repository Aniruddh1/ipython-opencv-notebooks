import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt

white = (255, 255, 255)
black = (0, 0, 0)
blue = (0, 0, 255)
navy = (128, 0, 0)
gray = (144, 128, 112)
green = (0, 255, 0)
red = (255, 0, 0)
yellow = (0, 255, 255)
magenta = (238, 0, 238)
orange = (0, 154, 238)

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

def polar2cart(r, theta_rad, centerXY):
    x = r  * np.cos(theta_rad) + centerXY[0]
    y = r  * np.sin(theta_rad) + centerXY[1]
    return x, y

def plot_imgs(img_data_lst, color=False, interp='none', max_cols=3, fig_size=10):
    cnt=len(img_data_lst)
    r,c,n = cnt,cnt,1
    for idx, img_data in enumerate(img_data_lst):
        if idx % max_cols == 0:
            plt.figure(figsize=(fig_size*r,fig_size*c))
        plt.subplot(r,c,idx+1)
        if color:
            #plt.imshow(img_data[0], interpolation='none', vmax=abs(img_data[0]).max(), vmin=-abs(img_data[0]).max())
            plt.imshow(img_data[0], interpolation=interp)
        else:
            plt.imshow(img_data[0], interpolation=interp, cmap='gray',vmin=0,vmax=255)
        plt.title('%s' % (img_data[1])), plt.xticks([]), plt.yticks([])

def create_pdf_of_plots(pdf_data):
    '''
    First create pdf_data dict, ie:
    
    pdf_data = {}
    pdf_data['pdf_file_name'] = 'NR2_ScorpionServer_data_plots.pdf'
    pdf_data['Title'] = u'NR2 ScorpionServer Data Plots'
    pdf_data['Author'] = u'Tom H. Rafferty / CNT'
    pdf_data['Subject'] = u'Data Plots For ScorpionServer image dumps'
    pdf_data['CreationDate'] = datetime.datetime.today()  # py3 uses datetime, py2 uses unicode str
    pdf_data['ModDate'] = datetime.datetime.today()       # py3 uses datetime, py2 uses unicode str

    pdf_data['plots_per_page'] = 5
    pdf_data['plot_data_lst'] = plot_data_lst   # list of tuples containing (np.array, plot_title_str)
    pdf_data['header_txt'] = 'NR2 ScorpionServer Image Data Plots\n(Flat lines means no data corruption)\n\n'
    pdf_data['xlabel'] = 'Dispense Row Number (blank lines skiped)'
    pdf_data['ylabel'] = 'Missing Drop\nIndication\n(0 = no missing)'
    
    Then call func, ie:
    
    create_pdf_of_plots(pdf_data)
    
    '''
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(pdf_data['pdf_file_name']) as pdf:
        # set the file's metadata via the PdfPages object:
        d = pdf.infodict() 
        d['Title'] = pdf_data['Title']
        d['Author'] = pdf_data['Author']
        d['Subject'] = pdf_data['Subject']
        d['CreationDate'] = pdf_data['CreationDate']  # py3 uses datetime, py2 uses unicode str
        d['ModDate'] = pdf_data['ModDate']       # py3 uses datetime, py2 uses unicode str

        # setup vars to control page layout
        num_plots = len(pdf_data['plot_data_lst'])
        plots_per_page = pdf_data['plots_per_page']
        num_pages = np.ceil(num_plots / plots_per_page)
        nb_pages = int(np.ceil(num_plots / float(plots_per_page)))
        grid_size = (plots_per_page, 1)

        print("Creating PDF (%s) with %d pages." % (pdf_data['pdf_file_name'], num_pages))
        print("Total plots: %d, plots per page: %d" % (num_plots, plots_per_page))

        for idx, data_to_plot in enumerate(pdf_data['plot_data_lst']):        
            # Create a figure instance (ie. a new page) if needed
            if idx % plots_per_page == 0:
                print("New page: Adding plot %d of %d" % (idx+1, len(pdf_data['plot_data_lst']))) 
                plt.figure(figsize=(8.27, 11.25), dpi=100)
                title = pdf_data['header_txt'] + data_to_plot[1]
            else:
                print("          Adding plot %d of %d" % (idx+1, len(pdf_data['plot_data_lst'])))
                title = data_to_plot[1]

            plt.subplot2grid(grid_size, (idx % plots_per_page, 0))
            plt.plot(data_to_plot[0], color='b')
            plt.title(title)
            plt.ylim([-(data_to_plot[0].max()+5), data_to_plot[0].max()+5])
            plt.xlabel(pdf_data['xlabel'])
            plt.ylabel(pdf_data['ylabel'])

            # Close the page if needed
            if (idx + 1) % plots_per_page == 0 or (idx + 1) == num_plots:
                plt.tight_layout()
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
                
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
            print("Image Engine initialized with %s" % (file_spec))
            print(" Total images: %d" % len(self.image_files))
        else:
            self.cap = cv2.VideoCapture(file_spec)
            print("Image Engine initialized with " % (file_spec))

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
