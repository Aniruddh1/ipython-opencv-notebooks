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
    parser.add_argument('-o', dest='output_spec', type=str, help='Output file spec (/path/to/folder/2015-08-04_process-*.png)')
    parser.set_defaults(video_file='')
    parser.set_defaults(glob_spec='')
    parser.set_defaults(output_spec='/tmp/process_img-*.png')
    parser.set_defaults(verbose=True)
    args = parser.parse_args()

    if len(args.glob_spec) > 0:
        eng = util.Image_Engine(args.glob_spec, is_glob=True)
    elif len(args.video_file) > 0:
        eng = util.Image_Engine(args.video_file, is_glob=False)
    else:
        print("\n\tError: invalid input: %s\n\n" % (args))
        parser.print_help()
        sys.exit(1)

    if len(args.output_spec) == 0:
        print("\n\tError: invalid input: %s\n\n" % (args))
        parser.print_help()
        sys.exit(1)

    ret = ip.createProcessImages_EdgeCircles(eng=eng, output_spec=args.output_spec)

    eng.cleanup()

if __name__ == "__main__":
    main()
