#!/usr/bin/env python

import sys
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
    parser.add_argument('-f', dest='video_file', type=str, help='Input file path (MP4, AVI, etc)')
    parser.add_argument('-g', dest='glob_spec',  type=str, help='Glob spec of image files to process (/path/to/folder/2015-08-04_DFI28-*.png)')
    parser.add_argument('-s', dest='strategy',  type=str, help='Strategy to use (supported: HoughCircle | CumSumDiff)')
    parser.add_argument('-d', dest='delay_s', type=float, default=0.020, help='Delay between frames in seconds (default: 0.02)')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Run in verbose mode (default: True)')
    parser.add_argument('--plot', dest='do_plot', action='store_true', help='Do plot data (default: True)')
    parser.add_argument('--no-plot', dest='do_plot', action='store_false', help='Do NOT plot data')
    parser.set_defaults(video_file='')
    parser.set_defaults(strategy='HoughCircle')
    parser.set_defaults(glob_spec='')
    parser.set_defaults(do_plot=True)
    parser.set_defaults(verbose=True)
    args = parser.parse_args()

    if len(args.glob_spec) > 0:
        eng = util.Image_Engine(args.glob_spec, is_glob=True)
        p, n = os.path.split(args.glob_spec)
        result_name = n.replace('-*', '').split('.')[0] + '_' + args.strategy
    elif len(args.video_file) > 0:
        eng = util.Image_Engine(args.video_file, is_glob=False)
        p, n = os.path.split(args.video_file)
        result_name = n.split('.')[0] + '_' + args.strategy
    else:
        print("\n\tError: invalid input: %s\n\n" % (args))
        parser.print_help()
        sys.exit(1)

    if args.strategy == 'HoughCircle':
        ret, stats = ip.processImages_HoughCircles(eng=eng, delay_s=args.delay_s, do_plot=args.do_plot, verbose=args.verbose)
    elif args.strategy == 'CumSumDiff':
        ret, stats = ip.processImages_CumSumDiff(eng=eng, delay_s=args.delay_s, do_plot=args.do_plot, verbose=args.verbose)
    else:
        print("\n\tError: Unsupported strategy: %s\n\n" % (args.strategy))
        parser.print_help()
        sys.exit(1)

    if not ret:
        print "!! Error processing images! ", result_name

    if 'area' in stats and len(stats['areas']) > 0:
        plt.figure(figsize=(10*2,5))
        plt.plot(stats['areas'])
        plt.title('Contact Areas')
        plt.savefig('../results/%s_ContactAreas.png' % (result_name))

    if 'center_pts' in stats and len(stats['center_pts']) > 0:
        plt.figure(figsize=(10*2,5))
        plt.plot([x for x,_ in stats['center_pts']])
        plt.plot([y for _,y in stats['center_pts']])
        plt.title('Center XY')
        plt.savefig('../results/%s_CenterXY.png' % (result_name))

    if 'center_pts' in stats and len(stats['center_pts']) > 0:
        plt.figure(figsize=(10*2,5))
        plt.plot([x for x,_ in stats['center_pts']], [y for _,y in stats['center_pts']])
        plt.title('Center Locs')
        plt.savefig('../results/%s_CenterLocs.png' % (result_name))

    if 'chan_means' in stats and len(stats['chan_means']) > 0:
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
