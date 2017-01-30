# -*- coding: utf-8 -*-
"""
Image processing utilities for Drop-In-Flight processing.

Created on Tue Feb  9 15:58:02 2016
@author: yesh
"""

import cv2
import pylab
import numpy as np
import logging

logger = logging.getLogger("blob_utils")

'''
Fit horizontal or vertical lines to points in ptList
ptList: Nx2 array of (row,col) (or (y,x) ... artifact of regionProps) 
        co-ordinates for N points
dimIdx   : 0 to fit horizontal lines, 1 to fit vertical lines
lineSpecPx: Wiggle room in pixels for point from line mean
idxList: List of indices in ptList to process. If None then all points in ptList
         are used. If empty list (=[]) then no elements are processed. 

Returns:
lines: list of lists. Each inner list is an index into ptList for points in a 
       single line
linesMean: Mean x (vert lines) or y (horiz lines) for each line.       
       
'''
def fitLines(ptList, dimIdx = 0, lineSpecPx = 25, idxList=None):
    idxListSpecified = True
    if (idxList == None):
        idxList = [i for i in range(len(ptList))]
        idxListSpecified = False
    numPts = len(idxList)
    if numPts == 0:
        logger.error("No blobs to fit line")
        return [],[]
    
    # Sort points by dimIdx
    sortedIdx = np.argsort(ptList[idxList,dimIdx]).tolist()
    if (idxListSpecified):
        sortedIdx = [idxList[sortedIdx[i]] for i in range(numPts)]

    # Add first point to its own line. Each line is a list of points in line.
    lines = [[sortedIdx[0]]]
    # Means along dimIdx for each line
    linesMean = [ptList[sortedIdx[0],dimIdx]]

    for i in range(1,numPts):
        currLineXorY = ptList[sortedIdx[i],dimIdx]
        if (currLineXorY > (linesMean[-1] + lineSpecPx)):
            # Add new line
            lines.append([sortedIdx[i]])
            linesMean.append(ptList[sortedIdx[i],dimIdx])            
        else:
            #print(lines)
            # Add drop to existing line and update mean
            numPtsInLine = len(lines[-1])
            linesMean[-1] = (linesMean[-1]*numPtsInLine + ptList[sortedIdx[i],dimIdx])/(numPtsInLine+1)
            lines[-1].append(sortedIdx[i])
            #lines[-1].insert(0,sortedIdx[i])
    
    return lines,linesMean
    
'''
Calculate average drop size in each line and overall
regProps : Region properties result for each blob centroid in ptList
lines: list of lists. Each inner list is an index into regProps for points in a 
       single line
medianFlag: If true calculates median of all drop sizes, else return -1

Return:
meanSizeInLine: List of mean drop sizes for each line 
overallMeanSize: Mean drop size of all drops
overallMedianSize: Median drop size of all drops
'''
def calcAvgDropSizePerLine(lines, regProps, medianFlag = False):
    numLines = len(lines)
    numPts = len(regProps)
    overallMeanSize = 0.0
    meanSizeInLine = np.zeros(numLines)
    if medianFlag:
        allDropSizes = np.zeros(numPts)
        dropCtr = 0;
    for i in range(numLines):
        numPtsInLine = len(lines[i])
        for j in range(numPtsInLine):
            meanSizeInLine[i] += regProps[lines[i][j]].area
            if medianFlag:
                allDropSizes[dropCtr] = regProps[lines[i][j]].area
                dropCtr += 1
        overallMeanSize += meanSizeInLine[i]
        meanSizeInLine[i] /= numPtsInLine
    overallMeanSize /= numPts
    if medianFlag:
        overallMedianSize = np.median(allDropSizes)
    else:
        overallMedianSize = -1;    
    return meanSizeInLine, overallMeanSize, overallMedianSize

'''
Prune blobs based on circularity and size.
Return:
filtIdx: Array with 0:numGoodBlobs containing indexes of good blobs in regProps
and numGoodBlobs:-1 containing indexes of bad blobs in regProps
numGoodBlobs: Number of good blobs
'''
def pruneBlobs(regProps, circTh=0.6, minSzInPx=50, maxSzInPx=300):
    numBlobs = len(regProps)
    # Good blob indices are 0:numGoodBlobs-1 and bad blob indices numGoodBlobs+1:-1
    filtIdx = np.zeros(numBlobs,dtype=int)
    
    numGoodBlobs = 0
    badBlobIdx = numBlobs-1
    for blobCtr in range(numBlobs):
        blobCirc = 0 if (regProps[blobCtr].major_axis_length < 1e-3) else regProps[blobCtr].minor_axis_length/regProps[blobCtr].major_axis_length
        if (regProps[blobCtr].area > minSzInPx) and (regProps[blobCtr].area < maxSzInPx) and (blobCirc > circTh):
            #print("i=%d:area = %f, circ = %f" % (blobCtr, regProps[blobCtr].area, blobCirc)) 
            filtIdx[numGoodBlobs] = blobCtr
            numGoodBlobs += 1
            #regPropsFilt = regPropsFilt.append(regProps[blobCtr])
        else:
            filtIdx[badBlobIdx] = blobCtr
            badBlobIdx -= 1;
    return filtIdx, numGoodBlobs

def preprocess_image(bg_img, fg_img, thresh_val, kernel_size):
    bg_img_hsv = cv2.cvtColor(bg_img, cv2.COLOR_BGR2HSV)
    fg_img_hsv = cv2.cvtColor(fg_img, cv2.COLOR_BGR2HSV)

    bg_img_hsv_split = cv2.split(bg_img_hsv)
    fg_img_hsv_split = cv2.split(fg_img_hsv)

    bg_img_diff = cv2.absdiff(bg_img_hsv_split[2], bg_img_hsv_split[0])
    fg_img_diff = cv2.absdiff(fg_img_hsv_split[2], fg_img_hsv_split[0])
#    pylab.figure()
#    pylab.imshow(fg_img_diff)
    pylab.figure()
    pylab.imshow(bg_img_hsv_split[2])
    img_delta = cv2.absdiff(bg_img_diff, fg_img_diff)

    ret,img_delta_thresh = cv2.threshold(img_delta, thresh_val, 255, cv2.THRESH_BINARY)
    #img_delta_thresh_blurred = cv2.GaussianBlur(img_delta_thresh, (kernel_size, kernel_size), 0)
    img_delta_thresh_blurred = cv2.medianBlur(img_delta_thresh, kernel_size, 0)
    return img_delta_thresh_blurred

'''
Preprocess image (img) and binarize it. The preprocessing step applies a 
blurKernelSize x blurKernelSize median filter to filter out salt-and-pepper
noise, and applies a top-hat transform with a kernel of size tophatKernalSize
to the inverted image to smooth the BG and enhance the blobs. The binirization
step applies a threshold calculated using method specified in thresholdType.

Return:
im_thresh: Binarized (aka thresholded) image  
'''
def preprocAndBinarizeImg(img, blurKernelSize = 5, tophatKernalSize = 21, minThVal = 0, maxThVal = 255, thresholdType = cv2.THRESH_BINARY+cv2.THRESH_OTSU):
    if len(img.shape) > 2:
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        im_gray = img
    im_gray = cv2.medianBlur(im_gray, blurKernelSize, 0)
    im_gray = cv2.morphologyEx(255-im_gray,cv2.MORPH_TOPHAT,np.ones([tophatKernalSize,tophatKernalSize]))
    #ret, im_thresh = cv2.adaptiveThreshold(im_gray,maxValue,adaptiveMethod, thresholdType,blockSize,subConst)
    ret, im_thresh = cv2.threshold(im_gray,minThVal,maxThVal,thresholdType)
    return im_thresh

# Difference-of-Gaussian + median filter + binarize
def simplePreProcAndBinarize(img):
    gauBlur1 = cv2.GaussianBlur(img,(3,3),0)
    gauBlur2 = cv2.GaussianBlur(img,(11,11),0)
    diffOfGau = gauBlur1-gauBlur2
    dog_med = cv2.medianBlur(diffOfGau, 9, 0)
    ret, im_thresh = cv2.threshold(dog_med,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return im_thresh

# Scale 16-bit image to 8-bit
def convert16BitTo8Bit(img16):
    # Scale ref to 8-bit
    minLevel = np.min(np.min(img16))
    img8 = (img16 - minLevel);
    maxLevel = np.max(np.max(img8))
    img8 = (img8*255.0/maxLevel).astype(np.uint8)
    return img8

'''
Match lines (vertical) of drops in reference image to corresponding lines of
drops in a target image. 
blobCentFiltRef : Nx2 array of (row,col) (or (y,x)) reference blob centroids
blobCentFiltTgt : Nx2 array of (row,col) (or (y,x)) target blob centroids
linesRef : Ref lines as a list of list of blob IDs for each line
linesTgt : Tgt lines as a list of list of blob IDs for each line
lineMeansRef : List of ref line means along stationary axis
lineMeansTgt : List of tgt line means along stationary axis
               All means are assumed to be sorted in ascending order
params : List of line spacing threshold params 
 [maxDiffInMeanForSameLines_px, maxInterDrpToDrpDist_px, maxDrpToDrpDistInTime_px]

Return:
CorresPts : Table of corresponding points. One row per set of corresponding 
points with columns [Ref blob ID, Tgt blob ID, Ref line #, Tgt line #, Ref drop X, Ref drop Y, Tgt drop X, Tgt drop Y]
nozzDisp_px : List of average drop displacement between ref and tgt drops for
each line (or nozzle)
nozzNumDrps : List of number of corresponding drops in each line (or nozzle)
corresLinesIdx : List ofcorresponding  (ref Line ID, tgt Line ID) pairs 
'''
def registerLines(blobCentFiltRef, blobCentFiltTgt, linesRef, linesTgt, lineMeansRef, lineMeansTgt, params):
    maxDiffInMeanForSameLines_px = params[0] 
    maxInterDrpToDrpDist_px = params[1]    # Should be set based on min possible velocity
    maxDrpToDrpDistInTime_px = params[2]

    tgtLineCtr = 0
    corresPts = []
    maxLines = min(len(lineMeansRef), len(lineMeansTgt))
    nozzDisp_px = np.zeros(maxLines)
    nozzNumDrps = np.zeros(maxLines)
    corresLinesIdx = []
    regLineCtr = -1     #Index counter for lines registered so far (0 => first line)
    for refLineCtr in range(len(lineMeansRef)):
        if (tgtLineCtr >= len(lineMeansTgt)):
            break
        if (abs(lineMeansRef[refLineCtr] - lineMeansTgt[tgtLineCtr]) < maxDiffInMeanForSameLines_px):
            # Corresponding lines found. Walk along the lines and find corres
            # points
            regLineCtr += 1
            corresLinesIdx.append([refLineCtr, tgtLineCtr])
            refPtCtr = 0
            tgtPtCtr = 0
            while ((refPtCtr < len(linesRef[refLineCtr])-1) and (tgtPtCtr < len(linesTgt[tgtLineCtr])-1)):
                refCurrPtIdx = linesRef[refLineCtr][refPtCtr]
                tgtCurrPtIdx = linesTgt[tgtLineCtr][tgtPtCtr]
                refNextPtIdx = linesRef[refLineCtr][refPtCtr+1]
                tgtNextPtIdx = linesTgt[tgtLineCtr][tgtPtCtr+1]
                currRefY = blobCentFiltRef[refCurrPtIdx][0]
                currTgtY = blobCentFiltTgt[tgtCurrPtIdx][0]
                nextRefY = blobCentFiltRef[refNextPtIdx][0]
                nextTgtY = blobCentFiltTgt[tgtNextPtIdx][0]
                # The tgt drop should ALWAYS be ahead of corresponding ref drop
                if ((currTgtY > currRefY) and (nextTgtY > nextRefY) and \
                (currTgtY - currRefY) < maxInterDrpToDrpDist_px) and \
                ((nextTgtY - nextRefY) < maxInterDrpToDrpDist_px) and \
                (abs(abs(nextRefY-currRefY)-abs(nextTgtY-currTgtY)) < maxDrpToDrpDistInTime_px):
                    if (nozzNumDrps[regLineCtr] == 0):
                        # Add current pt also
                        corresPts.append([refCurrPtIdx, tgtCurrPtIdx, refLineCtr, tgtLineCtr, blobCentFiltRef[refCurrPtIdx][1], currRefY, blobCentFiltTgt[tgtCurrPtIdx][1], currTgtY])
                        distTrav_px = pow(pow((corresPts[-1][4]-corresPts[-1][6]),2) + pow((corresPts[-1][5]-corresPts[-1][7]),2),0.5)
                        nozzDisp_px[regLineCtr] += distTrav_px
                        nozzNumDrps[regLineCtr] += 1
                    # Add next points as corresponding points
                    corresPts.append([refNextPtIdx, tgtNextPtIdx, refLineCtr, tgtLineCtr, blobCentFiltRef[refNextPtIdx][1], nextRefY, blobCentFiltTgt[tgtNextPtIdx][1], nextTgtY])
                    distTrav_px = pow(pow((corresPts[-1][4]-corresPts[-1][6]),2) + pow((corresPts[-1][5]-corresPts[-1][7]),2),0.5)
                    nozzDisp_px[regLineCtr] += distTrav_px
                    nozzNumDrps[regLineCtr] += 1                    
                    refPtCtr += 1
                    tgtPtCtr += 1
                elif ((currTgtY - currRefY) > maxInterDrpToDrpDist_px):
                    # Ref pt is behind tgt pt
                    refPtCtr += 1
                    continue
                elif ((currRefY - currTgtY) > maxInterDrpToDrpDist_px):
                    # Tgt pt is behind ref pt
                    tgtPtCtr += 1
                    continue
                else:
                    # TODO They are most likely sets of corres points but the
                    # dist between them are out of the defined threshold. For
                    # now just move on to next set of points
                    refPtCtr += 1
                    tgtPtCtr += 1
                    continue
                
            if (nozzNumDrps[regLineCtr] > 0):
                nozzDisp_px[regLineCtr] /= nozzNumDrps[regLineCtr]
            tgtLineCtr += 1
        elif ((lineMeansTgt[tgtLineCtr] - lineMeansRef[refLineCtr]) > maxDiffInMeanForSameLines_px):
            # Ref line is too far behind (left or top). Move to next ref line.
            continue
        elif ((lineMeansRef[refLineCtr] - lineMeansTgt[tgtLineCtr]) > maxDiffInMeanForSameLines_px):
            # Ref line is too far ahead (right or below). Find next good tgt line.
            tgtLineFound = False
            while (not(tgtLineFound) and (tgtLineCtr < len(lineMeansTgt)-1)):
                tgtLineCtr += 1
                if (abs(lineMeansRef[refLineCtr] - lineMeansTgt[tgtLineCtr]) < maxDiffInMeanForSameLines_px):
                    tgtLineFound = True
            if tgtLineFound:
                continue
            else:
                break
    # Trim the lists based on the actual number of lines registered
    nozzDisp_px = nozzDisp_px[0:regLineCtr+1]
    nozzNumDrps = nozzNumDrps[0:regLineCtr+1]    
    return corresPts, nozzDisp_px, nozzNumDrps, corresLinesIdx

'''
Calculate the average displacement (in pixels) of drops in each nozzle i.e.
drops in each vertical line. This differs from registerLines() in that it
assumes that the corresponding drops have already been registered. 
corresPts : Table (list of list) of corresponding points. One row per set of  
    corresponding points with columns 
    [Ref blob ID, Tgt blob ID, Ref line #, Tgt line #, Ref drop X, Ref drop Y, Tgt drop X, Tgt drop Y]  
linesRef : Ref lines as a list of list of blob IDs for each line
linesTgt : Tgt lines as a list of list of blob IDs for each line
lineMeansRef : List of ref line means along stationary axis
lineMeansTgt : List of tgt line means along stationary axis
               All means are assumed to be sorted in ascending order

Return:
nozzDisp_px : List of average drop displacement between ref and tgt drops for
each line (or nozzle)
nozzNumDrps : List of number of corresponding drops in each line (or nozzle) 
corresLinesIdx : List ofcorresponding  (ref Line ID, tgt Line ID) pairs
'''

def calcNozzDisplacement(corresPts, linesRef, linesTgt, lineMeansRef, lineMeansTgt): 
    
    # Create a dictionary of {ref blob ID, corresPt list} for faster lookup 
    refPtMap = {}
    for oneMatchingPt in corresPts:
        refPtMap[oneMatchingPt[0]] = oneMatchingPt
    
    # Create lookup of which line each tgt blob belongst o
    tgtBlobToLineMap = {}
    for i in range(len(linesTgt)):
        for j in range(len(linesTgt[i])):
            tgtBlobToLineMap[linesTgt[i][j]] = i
   
    numLines = len(lineMeansRef)
    nozzDisp_px = np.zeros(numLines)
    nozzNumDrps = np.zeros(numLines)
    corresLinesIdx = []

    for refLineCtr in range(len(lineMeansRef)):
        # FIXME - For now the tgt line corresponding to current ref line is just 
        # based off the first blob
        currCorresPt = refPtMap[linesRef[refLineCtr][0]] # corresPt of first blob in curr ref line
        tgtLineIdx = tgtBlobToLineMap[currCorresPt[1]]   # Tgt line corrsponding to above blob
        corresLinesIdx.append([refLineCtr, tgtLineIdx])
        # Walk the blobs in the ref line and calculate the average displacement
        # to the corresponding tgt 
        for refPtCtr in range(len(linesRef[refLineCtr])):
            currCorresPt = refPtMap[linesRef[refLineCtr][refPtCtr]]
            distTrav_px = pow(pow((currCorresPt[4]-currCorresPt[6]),2) + pow((currCorresPt[5]-currCorresPt[7]),2),0.5)
            nozzDisp_px[refLineCtr] += distTrav_px
            nozzNumDrps[refLineCtr] += 1        
                
        if (nozzNumDrps[refLineCtr] > 0):
            nozzDisp_px[refLineCtr] /= nozzNumDrps[refLineCtr]

    return nozzDisp_px, nozzNumDrps, corresLinesIdx    


'''
Register blobs from reference image (iRef) to target image (iTgt).
iRef            : Reference image or image ROI (tested with 8-bit)
iTgt            : Target image or image ROI
blobCentFiltRef : Nx2 array of (row,col) (or (y,x)) reference blob centroids
blobCentFiltTgt : Nx2 array of (row,col) (or (y,x)) target blob centroids
regPropsRefFilt : Region properties of reference blobs
regPropsTgtFilt : Region properties of target blobs
effPxSize_um    : Effective pixel size (width or height) in microns
blobRegParams   : Optional list of parameters 
    blobCorrBBXSzPx : Correlation bounding box X size in pixels
    blobCorrBBYSzPx : Correlation bounding box Y size in pixels
    blobCorrTh      : Correlation threshold in [0, 1]
    xDispPxTh       : X displacement threshold between corresponding blobs in pixels
    yDispPxTh       : Y displacement threshold between corresponding blobs in pixels
    areaDiffTh      : Blob area difference threshold between corresponding blobs in sq. px.
    majorAxisDiffTh : Blob major axis difference threshold between corresponding blobs in pixels
    minorAxisDiffTh : Blob minor axis difference threshold between corresponding blobs in pixels

Return:
CorresPts : Table of corresponding points. One row per set of corresponding 
points with columns [Ref blob ID, Tgt blob ID, Ref line #, Tgt line #, Ref drop X, Ref drop Y, Tgt drop X, Tgt drop Y, neighborhood correlation]
'''
def registerRefBlobsToTgt(iRef, iTgt, blobCentFiltRef, blobCentFiltTgt, regPropsRefFilt, regPropsTgtFilt, effPxSize_um, blobRegParams):
    # Parse the params for correlating blobs
    if (len(blobRegParams) > 0):
        blobCorrBBXSzPx = blobRegParams[0]
    else:
        blobCorrBBXSzPx = 101;

    if (len(blobRegParams) > 1):
        blobCorrBBYSzPx = blobRegParams[1]
    else:
        blobCorrBBYSzPx = 101;
    
    if (len(blobRegParams) > 2):
        blobCorrTh = blobRegParams[2]
    else:
        blobCorrTh = 0.95;    
    
    if (len(blobRegParams) > 3):
        xDispPxTh = blobRegParams[3]
    else:
        xDispPxTh = 10;
        
    if (len(blobRegParams) > 4):
        yDispPxTh = blobRegParams[4]
    else:
        yDispPxTh = 150;
    
    if (len(blobRegParams) > 5):
        areaDiffTh = blobRegParams[5]
    else:
        areaDiffTh = 100;
    
    if (len(blobRegParams) > 6):
        majorAxisDiffTh = blobRegParams[6]
    else:
        majorAxisDiffTh = 10;
    
    if (len(blobRegParams) > 7):
        minorAxisDiffTh = blobRegParams[7]
    else:
        minorAxisDiffTh = 10;
    
    numBlobsRefFilt = len(blobCentFiltRef)
    numBlobsTgtFilt = len(blobCentFiltTgt)
    # Register blobs from ref to tgt
    corresPts = []
    [numRows, numCols] = iRef.shape
    for i in range(numBlobsRefFilt):
        for j in range(numBlobsTgtFilt):
            # X displacement check
            if (abs(blobCentFiltTgt[j][1] - blobCentFiltRef[i][1]) > xDispPxTh):
                continue
            # Y displacement check
            if (abs(blobCentFiltTgt[j][0] - blobCentFiltRef[i][0]) > yDispPxTh):
                continue
            # If tgt blob is behind ref blob ... wrong match
            if ((blobCentFiltTgt[j][0] - blobCentFiltRef[i][0]) < 0):
                continue
            # TODO: Replace this naive test with something like cosine similarity
            # across a broader normalized feature vector
            if ((abs(regPropsTgtFilt[j].area - regPropsRefFilt[i].area) > areaDiffTh) or 
            (abs(regPropsTgtFilt[j].major_axis_length - regPropsRefFilt[i].major_axis_length) > majorAxisDiffTh) or 
            (abs(regPropsTgtFilt[j].minor_axis_length - regPropsRefFilt[i].minor_axis_length) > minorAxisDiffTh)):
                continue
            
            # Calculate correlation between the neighborhood
            # Check boundary and clip extents to make neighborhood sizes equal
            bbOffX = np.floor(blobCorrBBXSzPx/2);
            bbOffY = np.floor(blobCorrBBYSzPx/2);
              
            refCen = np.round(blobCentFiltRef[i]) 
            tgtCen = np.round(blobCentFiltTgt[j])
            
            bbOffX_left = np.min([refCen[1],tgtCen[1],bbOffX])
            bbOffX_right = np.min([numCols-refCen[1], numCols-tgtCen[1], bbOffX])        
            bbOffY_top = np.min([refCen[0],tgtCen[0],bbOffY])
            bbOffY_bot = np.min([numRows-refCen[0], numRows-tgtCen[0], bbOffY]) 
            
            refNN = iRef[refCen[0]-bbOffY_top:refCen[0]+bbOffY_bot, refCen[1]-bbOffX_left:refCen[1]+bbOffX_right]
            tgtNN = iTgt[tgtCen[0]-bbOffY_top:tgtCen[0]+bbOffY_bot, tgtCen[1]-bbOffX_left:tgtCen[1]+bbOffX_right]        
            
            corrRT = np.corrcoef(refNN.flatten(),tgtNN.flatten())  
            #print(i,j,corrRT)
            if (np.isfinite(corrRT[0][1]) and corrRT[0][1] < blobCorrTh):
                logger.warn("Found a similar size blob ... but neighborhood correlation(={0:3f}) < corrTh(={1:3f})".format(corrRT[0][1],blobCorrTh))
                continue
            
            # Check to see if the current ref blob has already been assigned 
            if ((len(corresPts) > 0) and (corresPts[-1][0] == i)):
                # If the new tgt blob has better correlation replace old match
                # TODO: Add a check based on drop velocity and position
                if (corrRT[0][1] > corresPts[-1][8]):
                    corresPts[-1] = [i, j, -1, -1, blobCentFiltRef[i][1], blobCentFiltRef[i][0], blobCentFiltTgt[j][1], blobCentFiltTgt[j][0], corrRT[0][1]]
            else:
                # Label as corresponding blobs
                corresPts.append([i, j, -1, -1, blobCentFiltRef[i][1], blobCentFiltRef[i][0], blobCentFiltTgt[j][1], blobCentFiltTgt[j][0], corrRT[0][1]])
            
#            distTrav_px = pow(pow((corresPts[-1][4]-corresPts[-1][6]),2) + pow((corresPts[-1][5]-corresPts[-1][7]),2),0.5)
            
#            # Calculate drop volumes from ref and tgt drops
#            dropArea_squm = regPropsRefFilt[i].area*effPxSize_um*effPxSize_um
#            dropVolRef_pL = 1e-3*4*np.pi*pow(pow(dropArea_squm/np.pi,0.5),3)/3
#            #print(i,': Ref vol (pL) = ',dropVolRef_pL)
#            
#            dropArea_squm = regPropsTgtFilt[j].area*effPxSize_um*effPxSize_um
#            dropVolTgt_pL = 1e-3*4*np.pi*pow(pow(dropArea_squm/np.pi,0.5),3)/3
#            #print(j,': Tgt vol (pL) = ',dropVolTgt_pL)
#    
#    for i in range(len(corresPts)):
#        print(i,': velocity (m/s) = ',(distTrav_px*rStep*pxSize_um)/timeLapse_us)
    
    return corresPts