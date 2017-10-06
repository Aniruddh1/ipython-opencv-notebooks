#!/usr/bin/env python

import os.path
import glob
import time
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

# local imports
import util
import ip_routines as ip

def main():
    '''
    '''
    parser = argparse.ArgumentParser(description='Contact Front estimation from video file or glob spec of image files')
    parser.add_argument('-i', dest='in_file', type=str, help='input file (MP4) or glob spec')
    parser.add_argument('-d', dest='delay_s', type=float, default=0.020, help='delay between frames')
    parser.add_argument('--is-glob', dest='is_glob', action='store_true', help='in_file is a glob spec')    
    parser.add_argument('--plot', dest='do_plot', action='store_true', help='Do plot data (default: True)')
    parser.add_argument('--no-plot', dest='do_plot', action='store_false', help='Do plot data (default: True)')
    parser.set_defaults(do_plot=True)
    parser.set_defaults(is_glob=False)
    args = parser.parse_args()

    verbose = True
    
    if args.in_file:
        eng = util.Image_Engine(args.in_file, is_glob=args.is_glob)
        
    if args.is_glob:
        p, n = os.path.split(args.in_file)
        result_name = n.replace('-*', '').split('.')[0]
    else:
        p, n = os.path.split(args.in_file)
        result_name = n.split('.')[0]

    ret, stats = ip.processImages_HoughCircles(eng, args.delay_s, args.do_plot, verbose)
    if not ret:
        print "!! Error processing images! ", args.in_file

    if len(stats['areas']) > 0:
        plt.figure(figsize=(10*2,5))
        plt.plot(stats['areas'])
        plt.title('Contact Areas')
        plt.savefig('../results/%s_ContactAreas.png' % (result_name))

    if len(stats['center_pts']) > 0:
        plt.figure(figsize=(10*2,5))
        plt.plot([x for x,_ in stats['center_pts']])
        plt.plot([y for _,y in stats['center_pts']])
        plt.title('Center XY')
        plt.savefig('../results/%s_CenterXY.png' % (result_name))

    if len(stats['center_pts']) > 0:
        plt.figure(figsize=(10*2,5))
        plt.plot([x for x,_ in stats['center_pts']], [y for _,y in stats['center_pts']])
        plt.title('Center Locs')
        plt.savefig('../results/%s_CenterLocs.png' % (result_name))

    if len(stats['chan_means']) > 0:
        plt.figure(figsize=(10*2,5))
        plt.plot([c for c,_,_ in stats['chan_means']], color='b')
        plt.plot([c for _,c,_ in stats['chan_means']], color='g')
        plt.plot([c for _,_,c in stats['chan_means']], color='r')
        plt.title('HSV mean values from center ROI')
        plt.savefig('../results/%s_HSV-Means.png' % (result_name))
    if args.do_plot: 
        plt.show()

    eng.cleanup()
    if args.do_plot: 
        while cv2.waitKey(1) & 0xFF != ord('q'):
            time.sleep(0.1)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
