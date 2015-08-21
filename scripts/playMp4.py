#!/usr/bin/env python
import numpy as np
import cv2
import time
import argparse

class Image_Engine:
    """ Image Engine
    """
    def __init__(self, file_spec, color=False):
        self.cap = cv2.VideoCapture(file_spec)
        self.image_idx = 0

    def next(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.image_idx += 1
                return True, frame
        else:
            return False, None
    
    def idx(self):
        return self.image_idx

    def cleanup(self):
        self.cap.release()

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

def process_frame(frame, idx):
    processed_frame = frame.copy()
    (rows, cols) = processed_frame.shape[:2]
    center = (rows/2, cols/2)
    roi_size = 100
    chan_means = calcChanMeans(processed_frame[center[1]-roi_size:center[1]+roi_size, center[0]-roi_size:center[0]+roi_size] )
    cv2.putText(processed_frame, ("%d" % (idx)), (center[0], center[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (255,0,0), 3)
    cv2.putText(processed_frame, ("R:%d, G:%d, B:%d" % (chan_means[2], chan_means[1], chan_means[0])), (int(center[0]), int(center[1])+100), cv2.FONT_HERSHEY_DUPLEX, 2, (255,0,0), 3)
    #processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    return processed_frame

if __name__ == "__main__":
    '''
    playMp4 -i path_to_mp4
    '''
    parser = argparse.ArgumentParser(description='Open MP4 and play it in window')
    parser.add_argument('-i', dest='in_file', type=str, help='input MP4 file')
    parser.add_argument('-d', dest='delay_s', type=int, help='delay between frames')
    args = parser.parse_args()

    if args.in_file:
        eng = Image_Engine(args.in_file)
        
        ret, frame = eng.next()
        while ret:
            processed_frame = process_frame(frame, eng.idx())
            
            cv2.imshow('frame',processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = eng.next()
    
        eng.cleanup()
        cv2.destroyAllWindows()
