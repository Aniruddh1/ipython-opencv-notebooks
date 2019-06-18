import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import queue # for CumSumDiff
from multiprocessing import Queue
from scipy import ndimage 

import util

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
    if verbose: print("  hypot: ", num)
    x, y = np.linspace(pt1[0], pt2[0], num), np.linspace(pt1[1], pt2[1], num)
    if verbose: print("pt1[0], pt2[0], x[0], x[-2], x[-1]:", pt1[0], pt2[0], x[0], x[-2], x[-1])
    if verbose: print("pt1[1], pt2[1], y[0], y[-2], y[-1]:", pt1[1], pt2[1], y[0], y[-2], y[-1])

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
    
    if verbose: print("  peakIdx, zi_diff[peakIdx] :", peakIdx, zi_diff[peakIdx])
    
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

def subimage2(image, centerXY, theta, width, height):
    #theta = np.deg2rad(theta)

    v_x = (np.cos(theta), np.sin(theta))
    v_y = (-np.sin(theta), np.cos(theta))
    s_x = centerXY[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
    s_y = centerXY[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])

    return cv2.warpAffine(image,mapping,(width, height),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)

def findPtsOnCircle(centerXY, r):
    angles = [0, np.pi/4.0, np.pi, np.pi/4.0*3, np.pi/4.0*5, np.pi/4.0*7]
    #angles = [np.pi/4.0, np.pi/4.0*3, np.pi/4.0*5, np.pi/4.0*7]
    pts = []
    for angle in angles:
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        pts.append((centerXY[0]+x, centerXY[1]+y))
    return pts

def findAvePt(pts):
    if len(pts) == 0: return (0,0)
    x=0
    y=0
    for pt in pts:
        x += pt[0]
        y += pt[1]
    return ( (x/len(pts), y/len(pts))  )

def findFirstPeakAboveThresh(vec, thresh):
    first_peak_idx = -99
    for idx, value in enumerate(vec):
        if value > thresh:
            first_peak_idx = idx
            #print("  idx_jump found: idx=%d, value=%d" % (idx, value))
            break
    return first_peak_idx
    
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
        if verbose: print("  threshold: ", delta_threshold)
        idx_jumps = []
        for idx, delta in enumerate(r):
            if delta > delta_threshold:
                idx_jumps.append( (idx, delta) )
                if verbose: print("  idx_jump found: idx=%d, delta=%d" % (idx, delta))
        if len(idx_jumps) > 0:
            peak_set = idx_jumps[0]
            if verbose: print("  peakIdx, delta:", peak_set[0], peak_set[1])
        else:
            peak_set = (-99, 0)
        peak_sets.append(peak_set)
    if verbose: print("  peaks:", peak_sets)

    lt_peakIdx = n/2 - peak_sets[0][0]
    rt_peakIdx = n/2 + peak_sets[1][0]
    if verbose: print("  >>> lt_peakIdx: %d, rt_peakIdx: %d" % (lt_peakIdx,rt_peakIdx))
    if verbose: print("  >>> nominal width: %d" % (rt_peakIdx-lt_peakIdx))
    return (rt_peakIdx-lt_peakIdx)

def estimateCircleDiameter(img):
    sample_halfWidth = 80
    # extract patch along center-width
    (rows, cols) = img.shape[:2]
    if verbose: print("  (estimateCircleDiameter) rows, cols: ", (rows, cols))
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

def estimateCircleDiameter2(img, centerXY, diag_len, verbose, debug_img, show_plots=False):
    #(rows, cols) = img.shape[:2]
    # pt = (center[1]+(376/2), center[0]-(488/2))
#     end_pt=(cols-1, 1)
#     offset=(0, 0)

#     x_len = abs(cols)
#     y_len = abs(rows)
#     hypot = int(np.hypot(x_len, y_len))
#     theta_rad = math.atan2(abs(y_len), abs(x_len))
#     #theta_rad = math.atan2((end_pt[1]-center_pt[1]), (end_pt[0]-center_pt[0]))
#     print("angle: %f" % np.rad2deg(theta_rad))
#     x,y = util.polar2cart(hypot/2, theta_rad, center_pt)
#     print "rot pt (center): ", (x,y)
#     x,y = abs(x+offset[0]), abs(y+offset[1])
#     print "rot pt (center) after offset: ", (x,y)

    if show_plots:
        plot_lst = []
        plot_lst.append( (img, 'img') )

    if True:
        patch_angles=[ (np.pi/4), (3*np.pi/4) ]  # these only work if cols == rows
    else:
        (rows, cols) = img.shape[:2]
        theta_rad = math.atan2(rows, cols)
        patch_angles=[ theta_rad, (np.pi-theta_rad) ]

    patch_width=70
    dia = []
    pt_sets = []
    for idx, patch_angle in enumerate(patch_angles):
        patch = subimage2(img, centerXY, patch_angle, patch_width, diag_len + (1*150))
        #patch_blur = cv2.GaussianBlur(patch, (15, 15), 0)
        patch_blur=patch
        patch_mean = np.abs(np.mean(patch_blur, axis=1))
        patch_diff = np.abs(np.diff(patch_mean))

        if show_plots:
            plot_lst.append( (patch_blur, ('patch %d' % (idx))) )
        
        pt1 = util.polar2cart(int(diag_len/2.0), patch_angle, centerXY)
        pt1 = ( int(pt1[0]), int(pt1[1]) )
        pt2 = util.polar2cart(int(diag_len/2.0), patch_angle + (4*np.pi/4), centerXY)
        pt2 = ( int(pt2[0]), int(pt2[1]) )
        #cv2.line(debug_img, pt1, pt2, (200, 255, 180), 2)
        angle_offsets = [np.pi/2, 6*np.pi/4]
        for angle_offset in angle_offsets:
            pt1_1 = util.polar2cart(int(patch_width/2.0), patch_angle+angle_offset, pt1)
            pt1_1 = ( int(pt1_1[0]), int(pt1_1[1]) )
            pt2_1 = util.polar2cart(int(patch_width/2.0), patch_angle + (4*np.pi/4)-angle_offset, pt2)
            pt2_1 = ( int(pt2_1[0]), int(pt2_1[1]) )
            cv2.line(debug_img, pt1_1, pt2_1, (100, 155, 180), 2)

        n = patch_diff.size
        ranges = []
        #~ lt_range = patch_diff[n/2::-1]
        #~ rt_range = patch_diff[n/2:]
        lt_range = patch_mean[n/2::-1]
        rt_range = patch_mean[n/2:]
        ranges.append(lt_range)
        ranges.append(rt_range)
        
        if show_plots:
            plt.figure(figsize=(10*2,10*2))
            plt.subplot(221)
            plt.plot(lt_range)
            plt.title("patch %d, left range (left half of patch_diff)" % (idx))
            plt.subplot(222)
            plt.plot(rt_range)
            plt.title("patch %d, right range (right half of patch_diff)" % (idx))

        # iter through each range, find the first peak past threshold
        peak_sets = []
        for r in ranges:
            delta_threshold = np.max(r) * 0.55
            if verbose: print("  (estimateCircleDiameter2) threshold: ", delta_threshold)
            idx_jumps = []
            for idx, delta in enumerate(r):
                if delta > delta_threshold:
                    idx_jumps.append( (idx, delta) )
                    #print("  idx_jump found: idx=%d, delta=%d" % (idx, delta))
            if len(idx_jumps) > 0:
                peak_set = idx_jumps[0]
                #print("  peakIdx, delta:", peak_set[0], peak_set[1])
            else:
                peak_set = (-99, 0)
            peak_sets.append(peak_set)
        #print("  peaks:", peak_sets)

        lt_peakIdx = n/2 - peak_sets[0][0]
        rt_peakIdx = n/2 + peak_sets[1][0]
        
        #~ print "  >>>>> rt_peakIdx, lt_peakIdx: ", peak_sets[1][0], peak_sets[0][0]
        #~ print "  >>>>> patch_angle, patch_angle2: ", patch_angle, patch_angle + (4*np.pi/4)
        pt1 = util.polar2cart(peak_sets[0][0], patch_angle, centerXY)
        pt1 = ( int(pt1[0]), int(pt1[1]) )
        pt2 = util.polar2cart(peak_sets[1][0], patch_angle + (4*np.pi/4), centerXY)
        pt2 = ( int(pt2[0]), int(pt2[1]) )
        pt_sets.append( (pt1, pt2) )
                
        #print(">>> lt_peakIdx: %d, rt_peakIdx: %d" % (lt_peakIdx,rt_peakIdx))
        if verbose: print("  (estimateCircleDiameter) nominal dia: %d" % (rt_peakIdx-lt_peakIdx))
        dia.append(rt_peakIdx-lt_peakIdx)

        if show_plots:
            util.plot_imgs(plot_lst)

#----- old:

    #~ patch1 = subimage2(img, centerXY, (np.pi/4), 60, diag_len + 150)
    #~ patch2 = subimage2(img, centerXY, (3*np.pi/4), 60, diag_len + 150)
    #~ #patch3 = subimage2(img, centerXY, (np.pi/2), 60, cols)
#~ 
    #~ patch1_blur = cv2.GaussianBlur(patch1, (15, 15), 0)
    #~ patch2_blur = cv2.GaussianBlur(patch2, (15, 15), 0)
    #~ #patch3_blur = cv2.GaussianBlur(patch3, (15, 15), 0)
#~ 
    #~ patch1_mean = np.abs(np.mean(patch1_blur, axis=1))
    #~ patch1_diff = np.abs(np.diff(patch1_mean))
    #~ patch2_mean = np.abs(np.mean(patch2_blur, axis=1))
    #~ patch2_diff = np.abs(np.diff(patch2_mean))
    #~ #patch3_mean = np.abs(np.mean(patch3_blur, axis=1))
    #~ #patch3_diff = np.abs(np.diff(patch3_mean))
#~ 
#~ 
    #~ patches = []
    #~ patches.append(patch1_diff)
    #~ patches.append(patch2_diff)
    #~ #patches.append(patch3_diff)
    #~ 
    #~ if show_plots:
        #~ util.plot_imgs([(img, 'img'), (patch1_blur, 'patch1'), (patch2_blur, 'patch2')], color=False)
#~ 
    #~ vector_sets=[  ]
    #~ dia=[]
    #~ for patch_diff in patches:
        #~ # split into two ranges: middle->left, middle->right
        #~ n = patch_diff.size
        #~ ranges = []
        #~ lt_range = patch_diff[n/2::-1]
        #~ rt_range = patch_diff[n/2:]
        #~ ranges.append(lt_range)
        #~ ranges.append(rt_range)
        #~ 
        #~ if show_plots:
            #~ plt.figure(figsize=(10*2,10*2))
            #~ plt.subplot(221)
            #~ plt.plot(lt_range)
            #~ plt.subplot(222)
            #~ plt.plot(rt_range)
#~ 
        #~ # iter through each range, find the first peak past threshold
        #~ peak_sets = []
        #~ for r in ranges:
            #~ delta_threshold = np.max(r) * 0.25
            #~ #if verbose: print "  (estimateCircleDiameter) threshold: ", delta_threshold
            #~ idx_jumps = []
            #~ for idx, delta in enumerate(r):
                #~ if delta > delta_threshold:
                    #~ idx_jumps.append( (idx, delta) )
                    #~ #print("  idx_jump found: idx=%d, delta=%d" % (idx, delta))
            #~ if len(idx_jumps) > 0:
                #~ peak_set = idx_jumps[0]
                #~ #print "  peakIdx, delta:", peak_set[0], peak_set[1]
            #~ else:
                #~ peak_set = (-99, 0)
            #~ peak_sets.append(peak_set)
        #~ #print "  peaks:", peak_sets
#~ 
        #~ lt_peakIdx = n/2 - peak_sets[0][0]
        #~ rt_peakIdx = n/2 + peak_sets[1][0]
        #~ 
        #~ vector_sets.append( (rt_peakIdx, lt_peakIdx) )
        #~ 
        #~ #print(">>> lt_peakIdx: %d, rt_peakIdx: %d" % (lt_peakIdx,rt_peakIdx))
        #~ if verbose: print("  (estimateCircleDiameter) nominal dia: %d" % (rt_peakIdx-lt_peakIdx))
        #~ dia.append(rt_peakIdx-lt_peakIdx)
    
    dia_np=np.array(dia)
    delta = np.abs(dia_np.mean() - diag_len)
    #percent_diff =  np.abs(diag_len - dia_np.mean()) / ((dia_np.mean() + diag_len)/2.0)
    
    if dia_np.std() > (dia_np.max() * 0.2):
        found = False
        print("  estimation failed1: %.2f > %2f" % (dia_np.std(), (dia_np.max() * 0.2)))
    #~ elif percent_diff > 0.1:
        #~ found = False
        #~ print("  estimation failed2: %.2f > 0.3" % (percent_diff))
        #~ print "   diag_len     : ", diag_len
        #~ print "   dia_np.mean(): ", dia_np.mean()
    else:
        found = True
        
    if not found:
        return -99, pt_sets 
    else:
        return dia_np.mean(), pt_sets

def estimateCircleDiameter3(img, centerXY, diag_len, verbose, debug_img, show_plots=False):
    (rows, cols) = img.shape[:2]

    if False:
        #patch_angles=[ (np.pi/4), (3*np.pi/4) ]  # these only work if cols == rows
        patch_angles=[ (np.pi/4), -(np.pi/4) ]  # these only work if cols == rows
    else:
        theta_rad = math.atan2(rows, cols)
        #patch_angles=[ theta_rad, (np.pi-theta_rad) ]
        #patch_angles=[ np.pi/2-theta_rad, -(np.pi/2-theta_rad) ]
        patch_angles=[ theta_rad, -(theta_rad) ]

    #patch_width=100
    patch_width=int(diag_len * 0.10)
    if patch_width > 100: patch_width = 100
    if patch_width < 25:  patch_width = 25

    dia = []
    pt_sets = []

    for idx, patch_angle in enumerate(patch_angles):
        print(">>> Patch idx: %d, angle: %.2f, diag_len: %d, patch_width: %d" % (idx, np.rad2deg(patch_angle), diag_len, patch_width))

        patch = subimage2(img, centerXY, (np.pi/2-patch_angle), patch_width, diag_len + patch_width)
        #patch_blur = cv2.GaussianBlur(patch, (15, 15), 0)
        patch_blur=patch
        patch_mean = np.abs(np.mean(patch_blur, axis=1))
        patch_diff = np.abs(np.diff(patch_mean))
        patch_values = patch_mean

        n = patch_values.size
        print(" patch length = %d" % (n))
        left_range  = patch_values[n/2::-1]
        right_range = patch_values[n/2:]

        left_peakIdx  = findFirstPeakAboveThresh(left_range,  (np.max(left_range) * 0.5) )
        right_peakIdx = findFirstPeakAboveThresh(right_range, (np.max(right_range) * 0.5) )

        left_edge  = n/2 - left_peakIdx
        right_edge = n/2 + right_peakIdx

        print(" left_peakIdx = %d,  left_edge = %d" % (left_peakIdx, left_edge))
        print(" right_peakIdx = %d, right_edge = %d" % (right_peakIdx, right_edge))
        #~ print "  >>>>> right_edge, left_edge: ", peak_sets[1][0], peak_sets[0][0]
        #~ print "  >>>>> patch_angle, patch_angle2: ", patch_angle, patch_angle + np.pi

        if left_peakIdx == -99 or right_peakIdx == -99:
            print(" Could not find peak...")
            #return -99, pt_sets

        if left_peakIdx == 0 or right_peakIdx == 0:
            print(" Could only find one peak...")
            #return -99, pt_sets

        pt1 = util.polar2cart(left_peakIdx, patch_angle, (centerXY[0], rows-centerXY[1]))
        pt1 = ( int(np.round(pt1[0])), int(rows-np.round(pt1[1])) )
        pt2 = util.polar2cart(right_peakIdx, patch_angle + np.pi, (centerXY[0], rows-centerXY[1]))
        pt2 = ( int(np.round(pt2[0])), int(rows-np.round(pt2[1])) )
        pt_sets.append( (pt1, pt2) )

        #print(">>> left_edge: %d, right_edge: %d" % (left_edge,right_edge))
        if verbose: print(" nominal dia: %d" % (right_edge-left_edge))
        dia.append(right_edge-left_edge)

        if show_plots:
            #
            # Rotate patch 90, then plot
            #
            pt1 = util.polar2cart(centerXY[0], patch_angles[1], (0,0))
            pt1 = ( int(pt1[0]), int(pt1[1]) )
            patch_rot = np.rot90(patch_blur, k=1).copy()
            cv2.line(patch_rot, (pt1[0], 0), (pt1[0], 70), 128, 3)
            plt.figure(figsize=(20,8))
            plt.subplot(111)
            plt.imshow(patch_rot, cmap='gray',vmin=0,vmax=255)
            plt.title(('patch %d' % (idx)))
            plt.xticks([t for t in range(50, patch_rot.shape[:2][1], 50)] )

            #
            # Calculate pts for patch width and add lines
            #
            pt1 = util.polar2cart(int(diag_len/2.0), patch_angle, centerXY)
            pt1 = ( int(pt1[0]), int(pt1[1]) )
            pt2 = util.polar2cart(int(diag_len/2.0), patch_angle + (4*np.pi/4), centerXY)
            pt2 = ( int(pt2[0]), int(pt2[1]) )
            #cv2.line(debug_img, pt1, pt2, (200, 255, 180), 2)
            angle_offsets = [np.pi/2, 6*np.pi/4]
            for angle_offset in angle_offsets:
                pt1_1 = util.polar2cart(int(patch_width/2.0), patch_angle+angle_offset, pt1)
                pt1_1 = ( int(pt1_1[0]), int(pt1_1[1]) )
                pt2_1 = util.polar2cart(int(patch_width/2.0), patch_angle + (4*np.pi/4)-angle_offset, pt2)
                pt2_1 = ( int(pt2_1[0]), int(pt2_1[1]) )
                cv2.line(debug_img, pt1_1, pt2_1, (100, 155, 180), 2)

            #
            # plot the left and right patch mean values separately
            #
            if False:
                plt.figure(figsize=(30,8))
                plt.subplot(221)
                plt.plot(left_range[::-1])
                plt.xticks([t for t in range(0, left_range.size, 50)] )
                plt.text(left_range.size-50, np.max(left_range)-50, ("%s" % (left_peakIdx)), fontsize=18)
                plt.title("patch %d, left range (left half of patch_mean)" % (idx))
                plt.subplot(222)
                plt.plot(right_range)
                plt.xticks([t for t in range(0, right_range.size, 50)] )
                plt.text(20, np.max(right_range)-50, ("%s" % (right_peakIdx)), fontsize=18)
                plt.title("patch %d, right range (right half of patch_mean)" % (idx))

            #
            # plot the patch_values
            #
            mid_pt = patch_values.size/2
            max_pt = np.max(patch_values)/2
            plt.figure(figsize=(50,6))
            plt.subplot(220 + idx + 1)
            plt.plot(patch_values)
            plt.vlines(left_edge, 0, max_pt + 25, color='r', linewidth=4)
            plt.text(left_edge-20, 20, ("%s" % left_edge), fontsize=12)
            plt.vlines(right_edge, 0, max_pt + 25, color='r', linewidth=4)
            plt.vlines(mid_pt, 0, max_pt, color='y', linewidth=2)
            plt.text(right_edge-20, 20, ("%s" % right_edge), fontsize=12)
            plt.text(left_edge+((right_edge-left_edge)/2), np.max(patch_values)-50, ("%s" % (right_edge-left_edge)), fontsize=18)
            plt.text(left_edge+((mid_pt-left_edge)/2), max_pt-50, ("%s" % (left_peakIdx)), fontsize=18)
            plt.text(mid_pt+((right_edge-mid_pt)/2), max_pt-50, ("%s" % (right_peakIdx)), fontsize=18)
            plt.title("patch %d, mean" % (idx))
            plt.xticks([t for t in range(50, patch_values.size, 50)] )
        
    dia_np=np.array(dia)
    delta = np.abs(dia_np.mean() - diag_len)
    #percent_diff =  np.abs(diag_len - dia_np.mean()) / ((dia_np.mean() + diag_len)/2.0)
    
    if dia_np.std() > (dia_np.max() * 0.2):
        found = False
        print("  estimation failed : %.2f > %2f" % (dia_np.std(), (dia_np.max() * 0.2)))
    else:
        found = True
        
    if not found:
        return -99, pt_sets 
    else:
        return dia_np.mean(), pt_sets
    
    
    

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
    if verbose: print("  (extractROI) startX, startY, endX, endY: ", startX, startY, endX, endY)
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

def findEllipse(img, seed_data, verbose=False, show_plots=False):
    debug_img = np.zeros(seed_data['img_dims'], np.uint8)
    debug_img[:,:,0] = img[:,:]
    debug_img[:,:,1] = img[:,:]
    debug_img[:,:,2] = img[:,:]
    
    centerYX  = seed_data['approx_center_yx']
    diag_len  = seed_data['approx_diameter'] 

    dia, pt_sets = estimateCircleDiameter3(img, (centerYX[1], centerYX[0]), diag_len, verbose, debug_img, show_plots)
    if dia == -99:
        print(" !! estimated cirle diameters (height vs width) too large")
        ellipse = ((-99, -99), (0,0), 0)
        return ellipse, debug_img
    
    if verbose: print(" estimated dia: ", dia)
#     width = findWidthOfCircle(img)
#     print("width: ", width)
    circle_pts = []
    for pt_set in pt_sets:
        print("pt_set: ", pt_set)
        cv2.line(debug_img, pt_set[0], pt_set[1], (0, 255, 0), 3)
        circle_pts.append(pt_set[0])
        circle_pts.append(pt_set[1])
    circle_fit_pts = np.array(circle_pts) 
    center, radius = cv2.minEnclosingCircle(circle_fit_pts)
    center = ( int(center[0]), int(center[1]) )
    radius = int(radius)
    cv2.circle(debug_img, center, radius, (0, 255, 255), 5)
    
    method = cv2.HOUGH_GRADIENT
    dp = 1           # inverse ratio of resolution
    min_dist = 10    # minimum distance between detected centers
    param1 = 60      # upper threshold for the internal Canny edge detector
    param2 = 30      # threshold for center detection.
    minRadius = int(dia/2)
    maxRadius = int(dia/2)+100
    #minRadius = int(dia/2) - int(int(dia/2) * 0.10)  
    #maxRadius = int(dia/2) + int(int(dia/2) * 0.20)

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT, dp, min_dist,
                               param1=param1, param2=param2, 
                               minRadius=minRadius, maxRadius=maxRadius)

    # circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=300,maxRadius=350)

    if circles is not None:
        numFound = len(circles[0,:])
        if verbose: print(" num circles found: ", numFound)


        # convert the (x, y) coordinates and radius of the circles to integers
        #circles = np.uint16(np.around(circles, 0))
        circles = np.round(circles[0, :]).astype("int")

        def getKey(item):
            return item[2]
        circles = sorted(circles, key=getKey)  # sort circles based on radius (item[2])

        pt_clusters=[ []  for x in xrange(7) ]
        pt_averages=[ ]

        if numFound <= 2:
            numToUse = numFound
        elif numFound > 2 and numFound <= 6:
            numToUse = int(numFound*0.8)
        elif numFound >= 15:
            numToUse = 15
        else:
            numToUse = int(numFound*0.6)
            
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles[0:numToUse]:
            #print("Center: ", (x,y), " Radius: ", r)
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
        #print("  >>> len(pt_clusters) = ", len(pt_clusters))
        for pt_cluster in pt_clusters:
            #print("    >>> len(pt_cluster) = ", len(pt_cluster))
            pt_averages.append(findAvePt(pt_cluster))
        #pt_averages = np.uint16(np.around(pt_averages, 0))
        #print pt_averages
        ellipse_fit_pts = np.array(pt_averages[1:])  
        ellipse = cv2.fitEllipse(ellipse_fit_pts) 
        #if verbose: print("Ellipse: ", ellipse)
        cv2.ellipse(debug_img,ellipse,(255,0,0),4)
        
        if seed_data['found_first']:
            new_diag_len = int(np.hypot(ellipse[1][0], ellipse[1][1]))
            if new_diag_len <= 0 or diag_len <= 0:
                print(" !! No more circles found...")
                ellipse = ((-99, -99), (0,0), 0)
                
            percent_diff =  (diag_len - new_diag_len) / ((new_diag_len + diag_len)/2.0)
            if percent_diff > 0.0:
                print("%sEllipse diag percent_diff getting smaller: %.3f (prev=%.2f, new=%.2f)%s" % (util.BLUE, percent_diff, diag_len, new_diag_len, util.RESET))
            elif percent_diff <= -0.1:
                print("%sEllipse diag percent_diff getting bigger BY TOO MUCH : %.3f (prev=%.2f, new=%.2f)%s" % (util.RED, percent_diff, diag_len, new_diag_len, util.RESET))
                ellipse = ((-99, -99), (0,0), 0)
            else:
                print("%sEllipse diag percent_diff getting bigger : %.3f (prev=%.2f, new=%.2f)%s" % (util.GREEN, percent_diff, diag_len, new_diag_len, util.RESET))
            
    else:
        print(" !! No circles found...")
        ellipse = ((-99, -99), (0,0), 0)

    return ellipse, debug_img

def processImages_HoughCircles(eng, delay_s, do_plot, verbose, capture_video_filename):
    stats = {}
    stats['areas'] = []
    stats['deltas'] = []
    stats['center_pts']=[]
    stats['chan_means'] = []
    previous_area = -1
    delta = 0
    
    thresh_val = 5
    kernel_size = 55

    ret, bg_img = eng.next()
    if not ret:
        return ret, stats

    bg_img_orig = bg_img.copy()
    bg_img = cv2.GaussianBlur(bg_img, (5, 5), 0)
    (rows, cols) = bg_img.shape[:2]
    stats['image_dims'] = (rows, cols)

    centerYX = (rows/2, cols/2)
    width = cols
    height = rows
    diag_len = int(np.hypot(rows, cols))

    cum_img_bool = np.zeros(bg_img.shape[:2], np.bool)
    total_size = cum_img_bool.size

    if do_plot: 
        print("Plotting is on!!")
        cv2.namedWindow('Images', cv2.WINDOW_AUTOSIZE)   #cv2.WINDOW_NORMAL

    seed_data = {}
    seed_data['found_first']  = False
    
    ret, fg_img = eng.next()
    while ret:
        img_idx = eng.idx()

        print("---------------------------------------------------------------------")
        print("FG img rows,cols:", fg_img.shape[:2])

        # let's first analyze a small ROI at the center point
        fg_img_converted = cv2.cvtColor(fg_img, cv2.COLOR_BGR2HSV)
        (rows, cols) = fg_img_converted.shape[:2]
        cp = (int(centerYX[0]), int(centerYX[1]))
        fg_img_converted_roi = fg_img_converted[cp[1]-10:cp[1]+10, cp[0]-10:cp[0]+10]  #np slice: [startY:endY, startX:endX]     
        chan_means = calcChanMeans(fg_img_converted_roi)
        stats['chan_means'].append( (chan_means[0], chan_means[1], chan_means[2]) )

        if True:
            delta_img_blurred = deltaImage(bg_img, fg_img, thresh_val)
            process_img = delta_img_blurred
        else:
            delta_image = cv2.absdiff(bg_img, fg_img)
            delta_image_gray = cv2.cvtColor(delta_image, cv2.COLOR_BGR2GRAY)
            ret,delta_image_thresh = cv2.threshold(delta_image_gray, thresh_val, 255, cv2.THRESH_BINARY)
            delta_img_blurred = cv2.GaussianBlur(delta_image_thresh, (kernel_size, kernel_size), 0)
            cum_img_bool = np.logical_or(cum_img_bool, delta_img_blurred)
            process_img = np.uint8(cum_img_bool) * 255

        print("%sImage index: [%d]%s" % (util.BLUE, img_idx, util.RESET))
        print("ROI center: (%d, %d), width: %d, height: %d" % (centerYX[0], centerYX[1], width, height))
        #process_img_roi = extractROI(process_img, (int(centerYX[0]), int(centerYX[1])), int(width), int(height))
        #print("  >> extracted img size: ", process_img_roi.shape[:2])

        seed_data['img_dims'] = (rows, cols, 3)
        seed_data['approx_center_yx'] = centerYX
        seed_data['approx_diameter']  = diag_len
        seed_data['diameter_tolerance']  = 0.20
         
        ellipse, debug_img = findEllipse(process_img, seed_data, verbose, False)
        
        if ellipse[0][0] == -99:
            print("%s Unable to find contact area, Image index: (%d)%s" % (util.RED, img_idx, util.RESET))
            stats['areas'].append(np.nan)
            stats['center_pts'].append( (np.nan, np.nan) )
            
            if True:
                #bg_img = bg_img_orig
                #bg_img = cv2.GaussianBlur(bg_img, (5, 5), 0)
                bg_img = cv2.GaussianBlur(fg_img, (5, 5), 0)
                (rows, cols) = bg_img.shape[:2]
                centerYX = (rows/2, cols/2)
                width = cols
                height = rows
                diag_len = int(np.hypot(rows, cols))
                        
            if do_plot: output = np.concatenate((fg_img, debug_img), axis=1)
            
            ret, fg_img = eng.next()

        else:
            seed_data['found_first']  = True

            print(ellipse)
            pad = 150
            #if ellipse[2] == 0:
            if True:
                plot_center = ((centerYX[0] + ellipse[0][1] - height/2), (centerYX[1] + ellipse[0][0] - width/2))
                centerYX = ((0*centerYX[0] + ellipse[0][1] - 0*height/2), (0*centerYX[1] + ellipse[0][0] - 0*width/2))
            elif ellipse[2] == 90:
                centerYX = ((0*centerYX[0] + ellipse[0][0] - 0*width/2), (0*centerYX[1] + ellipse[0][1] - 0*height/2))
    #         width = ellipse[1][0] + pad if (ellipse[1][0] + pad) <= cols else cols
    #         height = ellipse[1][1] + pad if (ellipse[1][1] + pad) <= rows else rows
            approx_area = (np.pi * np.hypot(ellipse[1][0]/2, ellipse[1][1]/2))
            
            #diag_len = int(np.max(ellipse[1]))
            diag_len = int(np.hypot(ellipse[1][0], ellipse[1][1]))

            stats['areas'].append(approx_area)
            stats['center_pts'].append( (centerYX[1], centerYX[0]) ) # append tuple
            if previous_area > 0:
                delta = approx_area-previous_area
                stats['deltas'].append(delta)
            previous_area = approx_area

            print("%s[%d] Area: %d, Delta: %d %s" % (util.GREEN, eng.idx(), approx_area, delta, util.RESET))

            if do_plot: 
                overlay = fg_img.copy()            
                #cv2.ellipse(overlay,(center, ellipse[1], ellipse[2]),(255,0,0),4) #cv2.ellipse(img, (centerX,centerY), (width,height), 0, 0, 180, color, lt)
                cv2.putText(overlay, ("%d"%(eng.idx())), (100, 100), cv2.FONT_HERSHEY_DUPLEX, 2, util.navy, 2)
                cv2.ellipse(overlay,((centerYX[1], centerYX[0]), ellipse[1], ellipse[2]),(255,0,0),4) #cv2.ellipse(img, (centerX,centerY), (width,height), 0, 0, 180, color, lt)
                cv2.putText(overlay, ("%d (%d)"%(approx_area, delta)), (int(centerYX[0])-200, int(centerYX[1])), cv2.FONT_HERSHEY_DUPLEX, 2, util.green, 3)
                output = np.concatenate((overlay, debug_img), axis=1)

            bg_img = cv2.GaussianBlur(fg_img, (5, 5), 0)
            ret, fg_img = eng.next()
            
        if do_plot: 
            output = cv2.resize(output, (0,0), fx=0.4, fy=0.4) 
            cv2.imshow('Images', output)
            if cv2.waitKey(int(delay_s*1000)) & 0xFF == ord('q'):
                break

    return True, stats

def processImages_EdgeCircles(eng, delay_s, do_plot, verbose, capture_video_filename):
    stats = {}
    stats['areas'] = []
    stats['deltas'] = []
    stats['center_pts']=[]
    previous_area = -1
    delta = 0
    
    img_lag_cnt = 5 # must be > 0
    img_queue = queue.Queue(img_lag_cnt)
    
    while img_queue.qsize() < img_queue.maxsize:
        ret, bg_img = eng.next()
        if not ret:
            print("Error: Not enough images to load queue")
            return ret, stats
        else:
            print("Adding image to queue...")
            img_queue.put(bg_img)

    (rows, cols) = bg_img.shape[:2]
    stats['image_dims'] = (rows, cols)

    centerXY = (cols/2, rows/2)
    diag_len = int(np.hypot(rows, cols))

    if do_plot: 
        print("Plotting is on!!")
        cv2.namedWindow('Images', cv2.WINDOW_AUTOSIZE)   #cv2.WINDOW_NORMAL

    if len(capture_video_filename) > 0:
        debug_img = np.zeros((rows, cols*2, 3), np.uint8)
        cv2.putText(debug_img, ("Start capture, press 's' when ready..."), (100, 200), cv2.FONT_HERSHEY_DUPLEX, 2, util.magenta, 2)
        output = np.concatenate((bg_img, debug_img), axis=1)
        cv2.imshow('Images', output)
        print("\n\nStart capture, press 's' when ready...")
        while not cv2.waitKey(1000) & 0xFF == ord('s'): pass

    cum_img_bool = np.zeros(bg_img.shape[:2], np.bool)
    total_size = cum_img_bool.size
    print("[%d] Background Area: %d" % (eng.idx(), total_size))

    mask_bool = np.empty(bg_img.shape[:2], np.uint8)
    mask_bool.fill(255)

    thresh_val = 8
    thresh_val = 10
    kernel_size = 55
    
    bg_img = img_queue.get()
    ret, fg_img = eng.next()
    while ret:
        print("---------------------------------------------------------------------")
        debug_img = np.zeros((rows, cols, 3), np.uint8)
        
        delta_image = cv2.absdiff(bg_img, fg_img)
        delta_image_gray = cv2.cvtColor(delta_image, cv2.COLOR_BGR2GRAY)
        ret,delta_image_thresh = cv2.threshold(delta_image_gray, thresh_val, 255, cv2.THRESH_BINARY)
        delta_img_blurred = cv2.GaussianBlur(delta_image_thresh, (kernel_size, kernel_size), 0)
        if True:
            cum_img_bool = np.logical_or(cum_img_bool, delta_img_blurred)
            process_img = np.uint8(cum_img_bool) * 255
        else:
            # this doesn't work...
            tmp_img = np.uint8(cum_img_bool) * 255
            process_img = cv2.bitwise_or(tmp_img, delta_img_blurred) 
            cum_img_bool = cv2.bitwise_and(process_img, mask_bool)
            cum_img_bool = cum_img_bool / 255.0
        
        if False:
            cv2.imwrite("../tmp/process_img_2015-08-10_PL9_ZB09_" + str(eng.idx()) + ".jpg", process_img)

        debug_img[:,:,0] = process_img[:,:]
        debug_img[:,:,1] = process_img[:,:]
        debug_img[:,:,2] = process_img[:,:]
        
        dia, pt_sets = estimateCircleDiameter3(process_img, centerXY, diag_len, verbose, debug_img, False)
        if verbose: print(" estimated dia: ", dia)

        if dia == -99:
            centerXY = (cols/2, rows/2)
            diag_len = int(np.hypot(rows, cols))

            print("%s[%d] Circle not found... %s" % (util.RED, eng.idx(), util.RESET))

            radius = 0
            area_text = "Area: n/a"

        else:
            circle_pts = []
            for pt_set in pt_sets:
                print("pt_set: ", pt_set)
                cv2.line(debug_img, pt_set[0], pt_set[1], (0, 255, 0), 3)
                circle_pts.append(pt_set[0])
                circle_pts.append(pt_set[1])
            circle_fit_pts = np.array(circle_pts) 
            (x,y), r = cv2.minEnclosingCircle(circle_fit_pts)
            
            #~ pts = findPtsOnCircle((x, y), r)
            #~ pts = np.float32(np.around(pts, 0))
            #~ ellipse_fit_pts = np.array(pts)
            #~ ellipse = cv2.fitEllipse(ellipse_fit_pts) 
            #~ cv2.ellipse(debug_img,ellipse,(255,0,0),4)
            
            centerXY = ( int(x), int(y) )
            radius = int(r)
            cv2.circle(debug_img, centerXY, radius, (0, 255, 255), 5)
            print("Center pt: ", centerXY, " radius: ", radius)

            diag_len = (2 * radius) + 150
            approx_area = (np.pi * (radius**2))

            stats['areas'].append(approx_area)
            stats['center_pts'].append( (centerXY[0], centerXY[1]) ) # append tuple
            if previous_area > 0:
                delta = approx_area-previous_area
                stats['deltas'].append(delta)
            previous_area = approx_area

            area_text = ("Area: %d (%d)"%(approx_area, delta))
            print("%s[%d] Area: %d, Delta: %d %s" % (util.GREEN, eng.idx(), approx_area, delta, util.RESET))

        if do_plot: 
            overlay = fg_img.copy()            
            #cv2.ellipse(overlay,(center, ellipse[1], ellipse[2]),(255,0,0),4) #cv2.ellipse(img, (centerX,centerY), (width,height), 0, 0, 180, color, lt)
            if radius > 0: cv2.circle(overlay, centerXY, radius, (255,0,0),4)
            cv2.putText(overlay, ("%d"%(eng.idx())), (100, 100), cv2.FONT_HERSHEY_DUPLEX, 2, util.navy, 2)
            cv2.putText(overlay, area_text, (int(centerXY[0])-200, int(centerXY[1])), cv2.FONT_HERSHEY_DUPLEX, 2, util.green, 3)
            output = np.concatenate((overlay, debug_img), axis=1)
            output = cv2.resize(output, (0,0), fx=0.4, fy=0.4) 
            cv2.imshow('Images', output)
            if delay_s > 0:
                if cv2.waitKey(int(delay_s*1000)) & 0xFF == ord('q'):
                    break
       
        img_queue.put(fg_img)
        bg_img = img_queue.get()
        #~ bg_img = fg_img
        ret, fg_img = eng.next()

    return True, stats

def processImages_CumSumDiff(eng, delay_s, do_plot, verbose, capture_video_filename):
    stats = {}
    stats['areas'] = []
    stats['deltas'] = []
    
    img_lag_cnt = 10 # must be > 0
    img_queue = queue.Queue(img_lag_cnt)
    
    while img_queue.qsize() < img_queue.maxsize:
        ret, bg_img = eng.next()
        if not ret:
            print("Error: Not enough images to load queue")
            return ret, stats
        else:
            print("Adding image to queue...")
            img_queue.put(bg_img)

    (rows, cols) = bg_img.shape[:2]
    stats['image_dims'] = (rows, cols)

    if do_plot: 
        print("Plotting is on!!")
        cv2.namedWindow('Images', cv2.WINDOW_AUTOSIZE)
    
    if len(capture_video_filename) > 0:
        #~ print "Creating output video: ", capture_video_filename
        #~ (width,height) = ( ((cols/2)*3), (rows/2) )
        #~ fourcc = cv2.VideoWriter_fourcc('p','i','m','1')
        #~ #fourcc = cv2.VideoWriter_fourcc('M','4','S','2')
        #~ #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 
        #~ cap_video = cv2.VideoWriter(capture_video_filename, -1, 30, (width,height), True)
        debug_img = np.zeros((rows, cols*2, 3), np.uint8)
        cv2.putText(debug_img, ("Start capture, press 's' when ready..."), (100, 200), cv2.FONT_HERSHEY_DUPLEX, 2, util.magenta, 2)
        output = np.concatenate((bg_img, debug_img), axis=1)
        cv2.imshow('Images', output)
        print("\n\nStart capture, press 's' when ready...")
        while not cv2.waitKey(1000) & 0xFF == ord('s'): pass

    cum_img_bool = np.zeros(bg_img.shape[:2], np.bool)
    total_size = cum_img_bool.size
    print("[%d] Background Area: %d" % (eng.idx(), total_size))

    thresh_val = 8
    kernel_size = 55
    
    bg_img = img_queue.get()
    ret, fg_img = eng.next()
    while ret:        
        print("---------------------------------------------------------------------")
        if False:
            delta_img = deltaImage(bg_img, fg_img, thresh_val)
            delta_img_blurred = cv2.GaussianBlur(delta_img, (kernel_size, kernel_size), 0)
        else:
            #bg_img = cv2.GaussianBlur(bg_img, (5, 5), 0)
            #bg_img_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
            
            #img = cv2.cvtColor(fg_img, cv2.COLOR_BGR2HSV)
            #chan = cv2.split(img)[0]
            #chan_blurred = cv2.medianBlur(chan, 5)

            #delta_image = cv2.absdiff(bg_img_gray, chan_blurred)
            
            delta_image = cv2.absdiff(bg_img, fg_img)
            
            delta_image_gray = cv2.cvtColor(delta_image, cv2.COLOR_BGR2GRAY)
            ret,delta_image_thresh = cv2.threshold(delta_image_gray, thresh_val, 255, cv2.THRESH_BINARY)
            delta_img_blurred = cv2.GaussianBlur(delta_image_thresh, (kernel_size, kernel_size), 0)


        before_cnt = np.count_nonzero(cum_img_bool)
        cum_img_bool = np.logical_or(cum_img_bool, delta_img_blurred)
        after_cnt = np.count_nonzero(cum_img_bool)

        area  = total_size - after_cnt
        delta = before_cnt - after_cnt 
        stats['areas'].append(area)
        stats['deltas'].append(delta)

        print("%s[%d] Area: %d, Delta: %d %s" % (util.GREEN, eng.idx(), area, delta, util.RESET))

        if do_plot: 
            overlay = fg_img.copy()
            cum_img = np.uint8(cum_img_bool) * 255
            debug_img = np.zeros((rows, cols*2, 3), np.uint8)
            debug_img[:,0:cols,0] = delta_img_blurred[:,:]
            debug_img[:,0:cols,1] = delta_img_blurred[:,:]
            debug_img[:,0:cols,2] = delta_img_blurred[:,:]
            #~ debug_img[:,0:cols,0] = delta_image[:,:,0]
            #~ debug_img[:,0:cols,1] = delta_image[:,:,1]
            #~ debug_img[:,0:cols,2] = delta_image[:,:,2]
            debug_img[:,cols:,0] = cum_img[:,:]
            debug_img[:,cols:,1] = cum_img[:,:]
            debug_img[:,cols:,2] = cum_img[:,:]
            centerXY = (cols/2, rows/2)
            radius = int(np.sqrt(area/np.pi))
            cv2.circle(overlay, centerXY, radius, (255,0,0), 4) #cv2.circle(img, centerXY, radius, color[, thickness[, lineType[, shift]]])
            cv2.putText(overlay, ("%d"%(eng.idx())), (100, 100), cv2.FONT_HERSHEY_DUPLEX, 2, util.navy, 2)
            cv2.putText(overlay, ("%d (%d)"%(area, delta)), (int(centerXY[0])-200, int(centerXY[1])), cv2.FONT_HERSHEY_DUPLEX, 2, util.green, 3)
            cv2.putText(debug_img, ("Thresh: %d, kernel: %d" % (thresh_val, kernel_size)), (50, 60), cv2.FONT_HERSHEY_DUPLEX, 2, util.blue, 2)
            cv2.putText(debug_img, ("Cumulative %%: %.3f" % (float(after_cnt)/float(total_size))), (cols+50, 60), cv2.FONT_HERSHEY_DUPLEX, 2, util.magenta, 2)
            output = np.concatenate((overlay, debug_img), axis=1)
            output = cv2.resize(output, (0,0), fx=0.4, fy=0.4) 
            cv2.imshow('Images', output)
            #cap_video.write(output)
            if cv2.waitKey(int(delay_s*1000)) & 0xFF == ord('q'):
                break
        
        img_queue.put(fg_img)
        bg_img = img_queue.get()
        #~ bg_img = fg_img
        ret, fg_img = eng.next()
        
    #~ if len(capture_video_filename) > 0:
        #~ cap_video.release()
    return True, stats

def createProcessImages_EdgeCircles(eng, output_spec):
    img_lag_cnt = 10 # must be > 0
    img_queue = Queue.Queue(img_lag_cnt)

    while img_queue.qsize() < img_queue.maxsize:
        ret, bg_img = eng.next()
        if not ret:
            print("Error: Not enough images to load queue")
            return ret, stats
        else:
            print("Adding image to queue...")
            img_queue.put(bg_img)

    cum_img_bool = np.zeros(bg_img.shape[:2], np.bool)
    total_size = cum_img_bool.size
    print("[%d] Background Area: %d" % (eng.idx(), total_size))

    mask_bool = np.empty(bg_img.shape[:2], np.uint8)
    mask_bool.fill(255)

    thresh_val = 8
    thresh_val = 6
    kernel_size = 25
    
    bg_img = img_queue.get()
    ret, fg_img = eng.next()
    while ret:
        outfile = output_spec.replace('*',  ('%04d' % eng.idx()))
        print("Creating file: " + outfile)       
        delta_image = cv2.absdiff(bg_img, fg_img)
        delta_image_gray = cv2.cvtColor(delta_image, cv2.COLOR_BGR2GRAY)
        ret,delta_image_thresh = cv2.threshold(delta_image_gray, thresh_val, 255, cv2.THRESH_BINARY)
        delta_img_blurred = cv2.GaussianBlur(delta_image_thresh, (kernel_size, kernel_size), 0)

        cum_img_bool = np.logical_or(cum_img_bool, delta_img_blurred)
        process_img = np.uint8(cum_img_bool) * 255

        cv2.imwrite( outfile, process_img )

        img_queue.put(fg_img)
        bg_img = img_queue.get()
        #~ bg_img = fg_img
        ret, fg_img = eng.next()

    return True

def faster_bradley_threshold(image, threshold=75, window_r=5): 
    percentage = threshold/100. 
    window_diam = 2*window_r + 1 
    
    foreground_value = 255
    img_dtype = np.uint8
    if image.dtype == 'uint16':
        foreground_value = 32768
        img_dtype = np.uint16
    
    # convert image to numpy array of grayscale values 
    img = image.astype(np.float) 

    # matrix of local means with scipy (this replaces integral image)
    means = ndimage.uniform_filter(img, window_diam) 
    
    # result: 0 for entry less than percentage*mean, 255 otherwise 
    height, width = img.shape[:2] 
    result = np.zeros((height,width), img_dtype) # initially all 0 
    result[img >= percentage * means] = foreground_value  # numpy magic :) 

    return result

def bradley_threshold(image, threshold=75, windowsize=5): 
    percentage = threshold/100. 
    window_diam = 2*windowsize + 1 
    
    foreground_value = 255
    img_dtype = np.uint8
    if image.dtype == 'uint16':
        foreground_value = 32768
        img_dtype = np.uint16
    
    # convert image to numpy array of grayscale values 
    img = image.astype(np.float) 

    # matrix of local means with scipy (this replaces integral image)
    means = ndimage.uniform_filter(img, window_diam) 
    
    # result: 0 for entry less than percentage*mean, 255 otherwise 
    height, width = img.shape[:2] 
    result = np.zeros((height,width), img_dtype) # initially all 0 
    result[img >= percentage * means] = foreground_value  # numpy magic :) 

    return result
