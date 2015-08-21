#!/usr/bin/env python

#~ get_ipython().magic(u'matplotlib')
#~ get_ipython().magic(u'load_ext autoreload')
#~ get_ipython().magic(u'autoreload 2')
import cv2
#import scipy.ndimage
import numpy as np
from matplotlib import pyplot as plt
import util
import glob

#from scipy import signal
import math
import collections

verbose = False

def deltaImage(bg_img, fg_img, thresh_val):
    bg_img = cv2.GaussianBlur(bg_img, (5, 5), 0)
    bg_img_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    #avg = bg_img.copy().astype("float")
    
    img = cv2.cvtColor(fg_img, cv2.COLOR_BGR2HSV)
    chan = cv2.split(img)[2]
    chan_blurred = cv2.medianBlur(chan, 5)

    delta_image = cv2.absdiff(bg_img_gray, chan_blurred)
    #img_thresh = cv2.adaptiveThreshold(delta_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 5)
    ret,delta_image_thresh = cv2.threshold(delta_image, thresh_val, 255, cv2.THRESH_BINARY)
    return delta_image_thresh

def findPeakIdx(img, pt1, pt2, verbose=False):
    x_len = abs(pt1[0]-pt2[0])
    y_len = abs(pt1[1]-pt2[1])
    num = int(np.hypot(x_len, y_len))
    if verbose: print "  hypot: ", num
    x, y = np.linspace(pt1[0], pt2[0], num), np.linspace(pt1[1], pt2[1], num)
    if verbose: print "pt1[0], pt2[0], x[0], x[-2], x[-1]:", pt1[0], pt2[0], x[0], x[-2], x[-1]
    if verbose: print "pt1[1], pt2[1], y[0], y[-2], y[-1]:", pt1[1], pt2[1], y[0], y[-2], y[-1]

    # Extract the values along the line, using cubic interpolation
    zi = scipy.ndimage.map_coordinates(img, np.vstack((y,x)))
    #print zi
    zi_int = zi.astype(int)
    zi_diff = np.abs(np.diff(zi_int))
    #print "len(zi_diff)=", len(zi_diff)

    # peakidx = signal.find_peaks_cwt(zi_diff, np.arange(1,2))
    # peakidx = np.argmax(zi_diff)
    
    n = 3
    idx_jumps = []
    for idx in range(n, zi_diff.size):
        pre_ave = np.average(zi_diff[idx-n:idx])
        post_ave = np.average(zi_diff[idx:idx+n])
        delta = post_ave - pre_ave
        if delta > 20:
            idx_jumps.append( (idx, delta) )
            if verbose: print("  idx_jump found: idx=%d, delta=%d" % (idx, delta))
    if len(idx_jumps) > 0:
        peakIdx = idx_jumps[0][0]
    else:
        peakIdx = np.argmax(zi_diff)
    
    if verbose: print "  peakIdx, zi_diff[peakIdx] :", peakIdx, zi_diff[peakIdx]
    
    return peakIdx, zi_diff, zi_int

def findPeakpt(img, pt1, pt2, verbose=False):
    peakIdx, zi_diff, zi_int = findPeakIdx(img, pt1, pt2, verbose)
    theta_rad = math.atan2(abs(pt2[1]-pt1[1]), abs(pt2[0]-pt1[0])) 
    x, y = util.polar2cart(peakIdx, theta_rad, pt1)
    return x, y

def rotateImage(image, angle):
    row,col = image.shape[:2]
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def subimage(img, centre, theta, width, height):
    #theta = np.deg2rad(theta)
    row,col = img.shape[:2]
    #output_image = cv.CreateImage((width, height), img.depth, img.nChannels)
    #output_image = np.zeros(img.shape, np.uint8)
    output_image = np.zeros((width, height), np.uint8)
    
    mapping = np.array([[np.cos(theta), -np.sin(theta), centre[0]],
                       [np.sin(theta), np.cos(theta), centre[1]]])
   
    #map_matrix_cv = cv.fromarray(mapping)
    map_matrix_cv = mapping
    
    #cv2.GetQuadrangleSubPix(img, output_image, map_matrix_cv)
    output_image = cv2.warpAffine(img, map_matrix_cv, (col, row)) 
    #output_image = cv2.warpAffine(img, map_matrix_cv, (width, height), output_image) 
    return output_image

def subimage2(image, center, theta, width, height):
    #theta = np.deg2rad(theta)

    v_x = (np.cos(theta), np.sin(theta))
    v_y = (-np.sin(theta), np.cos(theta))
    s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
    s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])

    return cv2.warpAffine(image,mapping,(width, height),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)

def findPtsOnCircle(center, r):
    angles = [0, np.pi/4.0, np.pi, np.pi/4.0*3, np.pi/4.0*5, np.pi/4.0*7]
    #angles = [np.pi/4.0, np.pi/4.0*3, np.pi/4.0*5, np.pi/4.0*7]
    pts = []
    for angle in angles:
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        pts.append((center[0]+x, center[1]+y))
    return pts

def findAvePt(pts):
    if len(pts) == 0: return (0,0)
    x=0
    y=0
    for pt in pts:
        x += pt[0]
        y += pt[1]
    return ( (x/len(pts), y/len(pts))  )

def findWidthOfCircle(img):
    # extract patch along center-width
    (rows, cols) = img.shape[:2]
    patch = img[rows/2-40:rows/2+40, 0:cols-1 ]  #np slice: [startY:endY, startX:endX]
    patch_blur = cv2.GaussianBlur(patch, (15, 15), 0)

    # find mean, diff
    patch_mean = np.abs(np.mean(patch, axis=0))
    patch_diff = np.abs(np.diff(patch_mean))

    # split into two ranges: middle->left, middle->right
    n = patch_diff.size
    ranges = []
    lt_range = patch_diff[n/2::-1]
    rt_range = patch_diff[n/2:]
    ranges.append(lt_range)
    ranges.append(rt_range)

    # iter through each range, find the first peak past threshold
    peak_sets = []
    for r in ranges:
        delta_threshold = np.max(r) * 0.25
        if verbose: print "  threshold: ", delta_threshold
        idx_jumps = []
        for idx, delta in enumerate(r):
            if delta > delta_threshold:
                idx_jumps.append( (idx, delta) )
                if verbose: print("  idx_jump found: idx=%d, delta=%d" % (idx, delta))
        if len(idx_jumps) > 0:
            peak_set = idx_jumps[0]
            if verbose: print "  peakIdx, delta:", peak_set[0], peak_set[1]
        else:
            peak_set = (-99, 0)
        peak_sets.append(peak_set)
    if verbose: print "  peaks:", peak_sets

    lt_peakIdx = n/2 - peak_sets[0][0]
    rt_peakIdx = n/2 + peak_sets[1][0]
    if verbose: print("  >>> lt_peakIdx: %d, rt_peakIdx: %d" % (lt_peakIdx,rt_peakIdx))
    if verbose: print("  >>> nominal width: %d" % (rt_peakIdx-lt_peakIdx))
    return (rt_peakIdx-lt_peakIdx)

def estimateCircleDiameter(img):
    sample_halfWidth = 80
    # extract patch along center-width
    (rows, cols) = img.shape[:2]
    if verbose: print "  (estimateCircleDiameter) rows, cols: ", (rows, cols)
    patch_w = img[rows/2-sample_halfWidth:rows/2+sample_halfWidth, 0:cols-1 ]  #np slice: [startY:endY, startX:endX]
    patch_w = cv2.GaussianBlur(patch_w, (15, 15), 0)

    patch_h = img[0:rows-1, cols/2-sample_halfWidth:cols/2+sample_halfWidth ]  #np slice: [startY:endY, startX:endX]
    patch_h = cv2.GaussianBlur(patch_h, (15, 15), 0)
    patch_h = np.rot90(patch_h)

    # find mean, diff
    patch_w_mean = np.abs(np.mean(patch_w, axis=0))
    patch_w_diff = np.abs(np.diff(patch_w_mean))
    patch_h_mean = np.abs(np.mean(patch_h, axis=0))
    patch_h_diff = np.abs(np.diff(patch_h_mean))

    patches = []
    patches.append(patch_w_diff)
    patches.append(patch_h_diff)
    
    dia=[]
    for patch_diff in patches:
        # split into two ranges: middle->left, middle->right
        n = patch_diff.size
        ranges = []
        lt_range = patch_diff[n/2::-1]
        rt_range = patch_diff[n/2:]
        ranges.append(lt_range)
        ranges.append(rt_range)

        # iter through each range, find the first peak past threshold
        peak_sets = []
        for r in ranges:
            delta_threshold = np.max(r) * 0.25
            #if verbose: print "  (estimateCircleDiameter) threshold: ", delta_threshold
            idx_jumps = []
            for idx, delta in enumerate(r):
                if delta > delta_threshold:
                    idx_jumps.append( (idx, delta) )
                    #print("  idx_jump found: idx=%d, delta=%d" % (idx, delta))
            if len(idx_jumps) > 0:
                peak_set = idx_jumps[0]
                #print "  peakIdx, delta:", peak_set[0], peak_set[1]
            else:
                peak_set = (-99, 0)
            peak_sets.append(peak_set)
        #print "  peaks:", peak_sets

        lt_peakIdx = n/2 - peak_sets[0][0]
        rt_peakIdx = n/2 + peak_sets[1][0]
        #print(">>> lt_peakIdx: %d, rt_peakIdx: %d" % (lt_peakIdx,rt_peakIdx))
        if verbose: print("  (estimateCircleDiameter) nominal dia: %d" % (rt_peakIdx-lt_peakIdx))
        dia.append(rt_peakIdx-lt_peakIdx)
    
    dia_np=np.array(dia)
    if dia_np.std() > 100:
        return -99
    else:
        return (dia_np.mean())

def estimateCircleDiameter2(img, center, diag_len, show_plots=False):
    #(rows, cols) = img.shape[:2]
    center_pt = (center[1], center[0])
    # pt = (center[1]+(376/2), center[0]-(488/2))
#     end_pt=(cols-1, 1)
#     offset=(0, 0)

#     x_len = abs(cols)
#     y_len = abs(rows)
    #hypot = int(np.hypot(x_len, y_len))
#     theta_rad = math.atan2(abs(y_len), abs(x_len))
#     #theta_rad = math.atan2((end_pt[1]-center_pt[1]), (end_pt[0]-center_pt[0]))
#     print("angle: %f" % np.rad2deg(theta_rad))
#     x,y = util.polar2cart(hypot/2, theta_rad, center_pt)
#     print "rot pt (center): ", (x,y)
#     x,y = abs(x+offset[0]), abs(y+offset[1])
#     print "rot pt (center) after offset: ", (x,y)

    patch1 = subimage2(img, center_pt, (np.pi/4), 60, diag_len)
    patch2 = subimage2(img, center_pt, (3*np.pi/4), 60, diag_len)
    #patch3 = subimage2(img, center_pt, (np.pi/2), 60, cols)

    patch1_blur = cv2.GaussianBlur(patch1, (15, 15), 0)
    patch2_blur = cv2.GaussianBlur(patch2, (15, 15), 0)
    #patch3_blur = cv2.GaussianBlur(patch3, (15, 15), 0)

    patch1_mean = np.abs(np.mean(patch1_blur, axis=1))
    patch1_diff = np.abs(np.diff(patch1_mean))
    patch2_mean = np.abs(np.mean(patch2_blur, axis=1))
    patch2_diff = np.abs(np.diff(patch2_mean))
    #patch3_mean = np.abs(np.mean(patch3_blur, axis=1))
    #patch3_diff = np.abs(np.diff(patch3_mean))


    patches = []
    patches.append(patch1_diff)
    patches.append(patch2_diff)
    #patches.append(patch3_diff)
    
    if show_plots:
        util.plot_imgs([(img, 'img'), (patch1_blur, 'patch1'), (patch2_blur, 'patch2')], color=False)

    dia=[]
    for patch_diff in patches:
        # split into two ranges: middle->left, middle->right
        n = patch_diff.size
        ranges = []
        lt_range = patch_diff[n/2::-1]
        rt_range = patch_diff[n/2:]
        ranges.append(lt_range)
        ranges.append(rt_range)
        
        if show_plots:
            plt.figure(figsize=(10*2,10*2))
            plt.subplot(221)
            plt.plot(lt_range)
            plt.subplot(222)
            plt.plot(rt_range)

        # iter through each range, find the first peak past threshold
        peak_sets = []
        for r in ranges:
            delta_threshold = np.max(r) * 0.25
            #if verbose: print "  (estimateCircleDiameter) threshold: ", delta_threshold
            idx_jumps = []
            for idx, delta in enumerate(r):
                if delta > delta_threshold:
                    idx_jumps.append( (idx, delta) )
                    #print("  idx_jump found: idx=%d, delta=%d" % (idx, delta))
            if len(idx_jumps) > 0:
                peak_set = idx_jumps[0]
                #print "  peakIdx, delta:", peak_set[0], peak_set[1]
            else:
                peak_set = (-99, 0)
            peak_sets.append(peak_set)
        #print "  peaks:", peak_sets

        lt_peakIdx = n/2 - peak_sets[0][0]
        rt_peakIdx = n/2 + peak_sets[1][0]
        #print(">>> lt_peakIdx: %d, rt_peakIdx: %d" % (lt_peakIdx,rt_peakIdx))
        if verbose: print("  (estimateCircleDiameter) nominal dia: %d" % (rt_peakIdx-lt_peakIdx))
        dia.append(rt_peakIdx-lt_peakIdx)
    
    dia_np=np.array(dia)
    if dia_np.std() > 100:
        return -99
    else:
        return (dia_np.mean())
    
def extractROI(img, center_pt, width, height):
    (rows, cols) = img.shape[:2]
    # do some clamping based on image extents 
#     startX = (center_pt[1] - width/2)  if (center_pt[1] - width/2) >= 0    else 0
#     startY = (center_pt[0] - height/2) if (center_pt[0] - height/2) >= 0   else 0
#     endX   = (center_pt[1] + width/2)  if (center_pt[1] + width/2) < cols  else cols-1
#     endY   = (center_pt[0] + height/2) if (center_pt[0] + height/2) < rows else rows-1
    startX = center_pt[1] - width/2 
    startY = center_pt[0] - height/2
    endX = center_pt[1] + width/2
    endY = center_pt[0] + height/2
    if verbose: print "  (extractROI) startX, startY, endX, endY: ", startX, startY, endX, endY
    return img[ startY:endY, startX:endX ]  #np slice: [startY:endY, startX:endX]

def calcChanMeans(img):
    chan1  = img.copy()
    chan1[:,:,1] = 0
    chan1[:,:,2] = 0
    chan1_mean = np.abs(np.mean(chan1))

    chan2 = img.copy()
    chan2[:,:,0] = 0
    chan2[:,:,2] = 0
    chan2_mean = np.abs(np.mean(chan2))

    chan3   = img.copy()
    chan3[:,:,0] = 0
    chan3[:,:,1] = 0
    chan3_mean = np.abs(np.mean(chan3))
    return (chan1_mean, chan2_mean, chan3_mean)

# ###Let's try the circle finding routines again...

# In[21]:

def findEllipse(img, center, diag_len, verbose=False, show_plots=False):
    debug_img = np.zeros((rows, cols, 3), np.uint8)

    dia = estimateCircleDiameter2(img, center, diag_len, show_plots)
    if dia == -99:
        print " !! estimated cirle diameters (height vs width) too large"
        ellipse = ((-99, -99), (0,0), 0)
        return ellipse, debug_img
    
    if verbose: print " estimated dia: ", dia
#     width = findWidthOfCircle(img)
#     print "width: ", width
    
    method = cv2.HOUGH_GRADIENT
    dp = 1           # inverse ratio of resolution
    min_dist = 20    # minimum distance between detected centers
    param1 = 60      # upper threshold for the internal Canny edge detector
    param2 = 30      # threshold for center detection.
    if False:
        minRadius = 300
        maxRadius = 340
    else:
        minRadius = int(dia/2)
        maxRadius = int(dia/2)+100 

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT, dp, min_dist,
                               param1=param1, param2=param2, 
                               minRadius=minRadius, maxRadius=maxRadius)

    # circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=300,maxRadius=350)

    if circles is not None:
        numFound = len(circles[0,:])
        if verbose: print " num circles found: ", numFound


        # convert the (x, y) coordinates and radius of the circles to integers
        #circles = np.uint16(np.around(circles, 0))
        circles = np.round(circles[0, :]).astype("int")

        #print circles
        def getKey(item):
            return item[2]
        circles = sorted(circles, key=getKey)
        #print circles

        pt_clusters=[ [], [], [], [], [], [], [] ]
        pt_averages=[  ]

        if numFound <= 2:
            numToUse = numFound
        elif numFound < 2 and numFound <= 6:
            numToUse = int(numFound*0.8)
        else:
            numToUse = int(numFound*0.2)
            
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles[0:numToUse]:
            #print "Center: ", (x,y), " Radius: ", r
            pts = findPtsOnCircle((x, y), r)
            pts = np.uint16(np.around(pts, 0))
            #print pts
            pt_clusters[0].append( (x,y) )
            for idx, pt in enumerate(pts):
                cv2.circle(debug_img, (pt[0],pt[1]), 5, (0,0,255),5)
                pt_clusters[idx+1].append( (pt[0],pt[1]) )

            # draw the outer circle
            cv2.circle(debug_img, (x, y), r, (0, 255, 0), 2)
            # draw the center of the circle
            if numFound <= 10:
                cv2.circle(debug_img,(x, y), 2, (0,0,255),3)
        #print pt_clusters
        #print "  >>> len(pt_clusters) = ", len(pt_clusters)
        for pt_cluster in pt_clusters:
            #print "    >>> len(pt_cluster) = ", len(pt_cluster)
            pt_averages.append(findAvePt(pt_cluster))
        #pt_averages = np.uint16(np.around(pt_averages, 0))
        #print pt_averages
        ellipse_fit_pts = np.array(pt_averages[1:])  
        ellipse = cv2.fitEllipse(ellipse_fit_pts) 
        #if verbose: print "Ellipse: ", ellipse
        cv2.ellipse(debug_img,ellipse,(255,0,0),4)
    else:
        print " !! No circles found..."
        ellipse = ((-99, -99), (0,0), 0)
    return ellipse, debug_img

plot_lst = []
verbose = True
thresh_val = 7  

#image_folder_lst = ['../projects/contact_front/MR_NewCam']
#image_folder_lst = ['../projects/contact_front/MR_NewCam/all']
image_folder_lst = ['../projects/contact_front/MR_NewCam/startAtFirstCircle']
#image_folder_lst = ['../projects/contact_front/MR_NewCam/transitionSet']

file_spec = '2015-08-04_DFI28-*.png'
glob_spec = "%s/%s" % (image_folder_lst[0], file_spec)
image_files = glob.glob(glob_spec)
image_files = sorted(image_files)

start_num = 1
#bg_img = eng.images[start_num-1].copy()
bg_img = cv2.imread(image_files[start_num-1], -1)



bg_img = cv2.GaussianBlur(bg_img, (5, 5), 0)
(rows, cols) = bg_img.shape[:2]
center = (rows/2, cols/2)
width = cols
height = rows
diag_len = int(np.hypot(rows, cols))

areas = []
chan1 = []
chan2 = []
chan3 = []
centerX=[]
centerY=[]
for img_idx, image_file in enumerate(image_files[start_num:]):
    print "---------------------------------------------------------------------"
    fg_img = cv2.imread(image_file, -1)
    print "FG img rows,cols:", fg_img.shape[:2]

    # let's first analyze a small ROI at the center point
    fg_img_converted = cv2.cvtColor(fg_img, cv2.COLOR_BGR2HSV)
    (rows, cols) = fg_img_converted.shape[:2]
    cp = (int(center[0]), int(center[1]))
    fg_img_converted_roi = fg_img_converted[cp[1]-10:cp[1]+10, cp[0]-10:cp[0]+10]  #np slice: [startY:endY, startX:endX]     
    chan_means = calcChanMeans(fg_img_converted_roi)
    chan1.append(chan_means[0])
    chan2.append(chan_means[1])
    chan3.append(chan_means[2])
    
    img_full = deltaImage(bg_img, fg_img, thresh_val)

    print("%s[%d]: (%s)%s" % (util.BLUE, img_idx, image_file, util.RESET))
    print("ROI center: (%d, %d), width: %d, height: %d" % (center[0], center[1], width, height))
    #img = extractROI(img_full, (int(center[0]), int(center[1])), int(width), int(height))
    img = img_full
    #print "  >> extracted img size: ", img.shape[:2]
    
    ellipse, output = findEllipse(img, center, diag_len, verbose)
    if ellipse[0][0] == -99:
        print("%s Unable to find contact area (%s)%s" % (util.RED, image_file, util.RESET))
        plot_lst.append( (img, 'findEllipse %d' % (img_idx) ))
        areas.append(np.nan)
        centerX.append(np.nan)
        centerY.append(np.nan)
        
        if True:
            bg_img = cv2.imread(image_files[start_num-1], -1)
            bg_img = cv2.GaussianBlur(bg_img, (5, 5), 0)
            (rows, cols) = bg_img.shape[:2]
            center = (rows/2, cols/2)
            width = cols
            height = rows

    else:        
        overlay = fg_img.copy()
        print ellipse
        pad = 150
        #if ellipse[2] == 0:
        if True:
            center = ((0*center[0] + ellipse[0][1] - 0*height/2), (0*center[1] + ellipse[0][0] - 0*width/2))
        elif ellipse[2] == 90:
            center = ((0*center[0] + ellipse[0][0] - 0*width/2), (0*center[1] + ellipse[0][1] - 0*height/2))
#         width = ellipse[1][0] + pad if (ellipse[1][0] + pad) <= cols else cols
#         height = ellipse[1][1] + pad if (ellipse[1][1] + pad) <= rows else rows
        approx_area = (np.pi * np.hypot(ellipse[1][0]/2, ellipse[1][1]/2))
        diag_len = int(np.max(ellipse[1]) + 150)
        areas.append(approx_area)
        centerX.append(center[1])
        centerY.append(center[0])

        print("%s Success! Contact area: %d %s" % (util.GREEN, approx_area, util.RESET))

        cv2.ellipse(overlay,(center, ellipse[1], ellipse[2]),(255,0,0),4) #cv2.ellipse(img, (centerX,centerY), (width,height), 0, 0, 180, color, lt)
        cv2.putText(overlay, ("%d"%(approx_area)), (int(center[0])-100, int(center[1])), cv2.FONT_HERSHEY_DUPLEX, 2, util.green, 3)
        plot_lst.append( (overlay, 'findEllipse %d' % (img_idx) ))
        bg_img = cv2.GaussianBlur(fg_img, (5, 5), 0)

plt.figure(figsize=(10*2,5))
plt.plot(areas)
# plt.vlines(lt_peakIdx, 0, np.max(patch_diff) + 25, color='r')
# plt.vlines(rt_peakIdx, 0, np.max(patch_diff) + 25, color='r')
plt.title('Contact Areas')

plt.figure(figsize=(10*2,5))
plt.plot(centerX)
plt.plot(centerY)
plt.title('Center points')

plt.figure(figsize=(10*2,5))
plt.plot(centerX, centerY)
plt.title('Center locs')

plt.figure(figsize=(10*2,5))
plt.plot(chan1[0:], color='b')
plt.plot(chan2[0:], color='g')
plt.plot(chan3[0:], color='r')
# plt.vlines(lt_peakIdx, 0, np.max(patch_diff) + 25, color='r')
# plt.vlines(rt_peakIdx, 0, np.max(patch_diff) + 25, color='r')
plt.title('HSV mean values from center ROI')
if False:
    plt.figure.max_num_figures=len(image_files) + 10
    plt.rcParams['figure.max_open_warning'] = 0
    util.plot_imgs(plot_lst)
