import cv2
import numpy as np
from matplotlib import pyplot as plt
import re
import json

import glob
import os


# ### Helper Functions

def readRaw(filepath, rows, cols):
    try: 
        with open(filepath, 'rb') as fd:
            raw_buf = np.fromfile(fd, dtype=np.uint8, count=rows*cols)
        #raw_buf = 255-raw_buf
        return raw_buf.reshape((rows, cols)) #notice row, column format
    except FileNotFoundError:
        print("Error! File not found: %s" % (filepath))
        raise
    
def readImage(path):
    ext = path[-3:]
    if ext == 'raw' or ext == 'dat':
        raw_rows = 1088
        raw_cols = 2048
        img = readRaw(path, raw_rows, raw_cols)
    if ext == 'bmp':
        img = cv2.imread(path)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
#         # get rid of 3rd channel
#         img = img[:,:,0]
#         # fix color map
#         img = 255-img
    return img
    
def plot_imgs(img_data_lst, color=False, interp='none', max_cols=3, fig_size=10):
    cnt=len(img_data_lst)
    r,c,n = cnt,cnt,1
    for idx, img_data in enumerate(img_data_lst):
        if idx % max_cols == 0:
            plt.figure(figsize=(fig_size*r,fig_size*c))
        plt.subplot(r,c,idx+1)
        if color:
            #plt.imshow(img_data[0], interpolation='none', vmax=abs(img_data[0]).max(), vmin=-abs(img_data[0]).max())
            plt.imshow(img_data[0].astype(np.float), interpolation=interp)
        else:
            plt.imshow(img_data[0].astype(np.float), interpolation=interp, cmap='gray')
        plt.title('%s' % (img_data[1])), plt.xticks([]), plt.yticks([])


folder_root = "/Users/trafferty/tmp/extrusion_images"


raw_rows = 1088
raw_cols = 2048


# ### Image Alignment


# Setup paths, exts

#img_root = "%s/request2_renamed" % (folder_root) 
img_root = "%s/NZ2-DS2/20180711-Longevity/Left-3p0" % (folder_root) 
GI_root = "%s/GoldenImages" % (img_root)

img_ext = 'bmp'
raw_ext = 'dat'


# First build a map identifying the Golden Image field number for each location

GI_files = glob.glob("%s/**/*.%s" % (GI_root, img_ext))



GI_map = {}
for f in GI_files:
    file_dir, file_name = os.path.split(f)

    fname = file_name.split('.')[0]
    d = fname.split('_')
    #print(d)
    GI_map["%s_%s" % (d[2], d[3])] = d[1] 

print("Golden image map:", GI_map)

imgFolders = glob.glob("%s/*/" % img_root)
print("img_root:\n\t", img_root)
print("imgFolders:\n\t", imgFolders)

def getGoldenImage(locStr):
    if locStr not in GI_map:
        return ""
    f = GI_map[locStr]
    GI = "Field_%s_%s.%s" % (f, locStr, img_ext)
    return GI

locs = []

for imgFolder in imgFolders:
    subFolder = imgFolder.split('/')[-2]
    
    try:
        int(subFolder.lower().split('_')[0])
        locs.append(subFolder)
    except ValueError:
        pass
    
    GI = getGoldenImage(subFolder)
    if len(GI) == 0:
        print("No GI for location: ", subFolder)
        continue
    print(GI)

print(locs)


def findAlignMatrix(unaligned_img, target_img, warp_mode, max_iters):
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iters,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (target_img, unaligned_img, warp_matrix, warp_mode, criteria)

    return warp_matrix
    
def alignImage(unaligned_img, target_img, warp_mode, warp_matrix):
    sz = unaligned_img.shape

    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        aligned_img = cv2.warpPerspective (unaligned_img, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        aligned_img = cv2.warpAffine(unaligned_img, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        
    return aligned_img



loc = locs[0]
print("loc: ", loc)
loc_path = "%s/%s" % ( img_root, loc)
loc_files = glob.glob("%s/*.%s" % (loc_path, img_ext))
print(len(loc_files))

GI_path = "%s/%s" % ( loc_path, getGoldenImage(loc))
print("GI_path:", GI_path)

GI_img =readImage( GI_path)


number_of_iterations = 50;
all_norms = {}

for loc in locs:
    loc_path = "%s/%s" % ( img_root, loc)
    loc_files = glob.glob("%s/*.%s" % (loc_path, img_ext))
    print("Processing loc: ", loc_path)

    GI = getGoldenImage(loc)
    if len(GI) == 0:
        print("No GI for location: ", loc)
        continue

    GI_path = "%s/%s" % ( loc_path, GI)
    print("GI_path:", GI_path)

    GI_img =readImage( GI_path)

    norms = []
    for idx, loc_file in enumerate(loc_files):
        print("Processing image (%d of %d): %s" % (idx, len(loc_files), loc_file))
        loc_img = readImage(loc_file)

        warp_mode = cv2.MOTION_EUCLIDEAN
        #warp_mode = cv2.MOTION_TRANSLATION
        
        warp_matrix_EUCL = findAlignMatrix(loc_img, GI_img, warp_mode, number_of_iterations)

        norm = np.linalg.norm(warp_matrix_EUCL,ord=2)
        norms.append(str(norm))
        #print("norm of matrix: ", norm)
        print(norm)
        print(warp_matrix_EUCL)
    all_norms[loc] = norms


# open output file for writing
with open('all_norms.json', 'w') as f:  
    json.dump(all_norms, f)