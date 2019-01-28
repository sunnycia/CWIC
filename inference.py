import numpy as np
import caffe
import cv2
import math

from utils import *

caffe.set_device(0)
caffe.set_mode_gpu()
# net=caffe.Net('prototxt/cwic_deploy.prototxt','/data/sunnycia/image_compression_challenge/_Train/ImageCompression/model/cmp/5.caffemodel',caffe.TEST)
net=caffe.Net('prototxt/cwic_deploy.prototxt','cwic_training_output_fixw0001_ADAM/snapshot-_iter_60000.caffemodel',caffe.TEST)

output_channel=64

ori_image_path = 'kodim16.png'

img = cv2.imread(ori_image_path)
# print img.shape;exit()

if img.shape[0] % 8 >0:
   img=img[0:img.shape[0]-img.shape[0]%8,:]
if img.shape[1] % 8 >0:
   img=img[:,0:img.shape[1]-img.shape[1]%8]
net.blobs['data'].reshape(1,3,img.shape[0],img.shape[1])

# data=(img-127.5)/127.5
# net.blobs['data'].data[0]=data
data=(img.transpose(2,0,1)-127.5)/127.5
net.blobs['data'].data[...] = data
net.forward()
pimg=net.blobs['imp_conv2'].data[0,0]*255
gimg=net.blobs['pdata'].data[0]*127.5+127.5

bimp = net.blobs['imap'].data[...]
print bimp.shape
print pimg.shape
ori_size = output_channel*pimg.shape[0]*pimg.shape[1]
cmp_size = bimp.sum()

gimg[gimg<0]=0
gimg[gimg>255]=255
gimg=gimg.transpose(1,2,0).astype(np.uint8)

print gimg.shape

trimg=img.transpose(2,0,1)
trorg=gimg.transpose(2,0,1)
yimg=ytrans(trimg)
yorg=ytrans(trorg)

cv2.imwrite('kodim16-reconst.png', gimg)
cv2.imwrite('kodim16-imp.png', pimg)
print 'Y-psnr:'+str(psnr(yimg,yorg))
print ori_size, cmp_size, cmp_size/ori_size