import math
import numpy as np
from math import ceil
import random
import cv2
import glob
# from PIL import Image
import imghdr
import os
import random

psnr=lambda x,y:10*math.log10(255.0*255.0/(np.sum(np.square(y.astype(np.float)-x))/float(x.size)))
ytrans=lambda x:0.299*x[2]+0.587*x[1]+0.114*x[0]

class binary_encoder:
    '''
    The encoder part is the copy from the PAQ coder which is a application of arithmetic coding.
    '''
    def __init__(self):
        self.x1=0
        self.x2=0xffffffff
    def restart(self):
        self.x1=0
        self.x2=0xffffffff
    def coding_bit(self,ch,mp):
        p = int(mp*65535.0)
        xdiff=self.x2-self.x1
        xmid=self.x1
        if xdiff>=0x10000000: xmid+=(xdiff>>16)*p
        elif xdiff>=0x1000000: xmid+=((xdiff>>12)*p)>>4
        elif xdiff>=0x100000: xmid+=((xdiff>>8)*p)>>8
        elif xdiff>=0x10000: xmid+=((xdiff>>4)*p)>>12
        else: xmid+=(xdiff*p)>>16
        if ch>0:self.x1=xmid+1
        else:self.x2=xmid
        l=0
        while ((self.x1^self.x2)&0xff000000)==0:
            l+=8
            self.x1=(self.x1<<8)&0xffffffff
            self.x2=((self.x2<<8)+255)&0xffffffff
        # print 'l:', l
        return l

def check_prime(number): 
    # if not type(number)==int:
        # print "Please input an interger number."
        # return 0;
    ceil = int(np.sqrt(number));
    for i in range(2, ceil+1):
        if number%i == 0:
            return 0
    return 1
            
def explode_number(number):
    if not type(number)==int:
        print "Please input an interger number."
        return 0;
    while check_prime(number):
        print "It's a prime"
        number += 1
    a = int(np.sqrt(number))
    if a**2 == number:
        return a, a
    while not number%a == 0:
        a -= 1
    b = number /a
    if a > b:
        b, a = a, b
    return a, b
def jigsaw(imageDir, output_path=None, stdsize=(30, 30), padding=0, mode=cv2.IMREAD_COLOR, rdm_portion=1, arrange=0):
    imgs = []
    delta=0
    imageList = os.listdir(imageDir)
    # random.shuffle(imageList)
    # imageList.sort()
    imageList.sort
    
    file_num = len(imageList)
    for filename in imageList:
        imagePath = os.path.join(imageDir, filename)
        ## check if an image
        imgType = imghdr.what(imagePath)
        if imgType==None:
            print imagePath, "is not an regular Image file"
            file_num -= 1
            continue
        
        img_arr = cv2.imread(imagePath, mode)
        img_arr = cv2.resize(img_arr, stdsize)
        imgs.append(img_arr)
    
    if arrange==0:
        row, col = explode_number(file_num)
    else:
        row = arrange[0]
        col = arrange[1]
    print row, col;
    if row*col > file_num:
        delta =  row*col - file_num
        if mode == cv2.IMREAD_GRAYSCALE:
            patch_img = np.ones(stdsize)
        else:
            patch_img = np.ones((stdsize[1], stdsize[0], 3))
        patch_img *= 255
        for i in range(delta):
            imgs.append(patch_img)
    print imgs[0].shape, imgs[-1].shape
    img = np.concatenate(imgs, 0)
    # cv2.imwrite("damn.jpg", img);exit()
    print img.shape
    if mode == cv2.IMREAD_GRAYSCALE:
        img = img.reshape(row, col, stdsize[0], stdsize[1])
        img = img.swapaxes(1, 2).reshape(row*stdsize[0], col*stdsize[1])
    else:
        img = img.reshape(row, col, stdsize[1], stdsize[0], 3)
        print img.shape
        if not padding==0:
            mask = np.ones(img.shape[:-1], dtype=bool)
            mask[:, :, padding:-padding, padding:-padding]=False
            img[mask] = 255
        
        img = img.swapaxes(1, 2).reshape(row*stdsize[1], col*stdsize[0], 3)
        print img.shape
    
    if output_path == None:
        cv2.imwrite(imageDir+".jpg", img)
    else:
        cv2.imwrite(output_path, img)