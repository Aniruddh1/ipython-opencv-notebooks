# -*- coding: utf-8 -*-
"""
An imitation of Octave's pixval() functionality that is used to display
(x,y,pixel value) as you move the mouse over an image.

Example usage:
fig, ax = pylab.subplots()
img_pixval = ax.imshow(img, cmap = 'gray', interpolation='none')
ax.format_coord = Formatter(img_pixval)

Created on Thu Jan 21 14:53:12 2016

@author: yesh
"""

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)

