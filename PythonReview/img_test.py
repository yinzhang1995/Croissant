#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:44:18 2020

@author: legendary_yin
"""

# dealing with image

from PIL import Image
import os
import numpy as np

os.getcwd()
os.chdir('./Image/')
os.getcwd()

fgimg  = Image.open('myImage.png')
bgimg  = Image.open('myImage (1).png')


fgimg.show()
bgimg.show()

fgimg.size
bgimg.size

pixfg = fgimg.load()
pixbg = bgimg.load()

for i in range(fgimg.size[0]):
    for j in range(fgimg.size[1]):
        if pixfg[i,j] == (0, 255, 0, 255):
            pixfg[i,j] = pixbg[i,j]
            

fgimg.show()
fgimg.save('output.png')


## smaller the image
im = Image.new(mode = "RGB", size = (192, 108))
im.show()

imgmap = im.load()
for i in range(0,im.size[0]):
    for j in range(0,im.size[1]):
        imgmap[i,j] = pixfg[i * 10,j * 10]
        
im.show()


## flatten the image
#'C’ means to flatten in row-major (C-style) order. ‘F’ means to flatten in column-major (Fortran- style) order
#a.flatten(order = 'F')

m = []
for i in range(0,im.size[0]):
    for j in range(0,im.size[1]):
        m.append(imgmap[i,j])
        
imgmap
