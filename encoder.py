import cv2
import caffe
import os
from utils import binary_encoder


class Encoder:

    def __init__(self, quality_index=1, model_dir='compression_model', deploy_prototxt_path='prototxt/64channel_encoder.prototxt'):
        self.model_path = os.path.join(model_dir, '/%s.caffemodel' % str(quality_index))
        self.deploy_path = deploy_prototxt_path
        if not os.path.isfile(self.model_path):
            print self.model_path, "not exists, abort."
            exit()

        if not os.path.isfile(self.deploy_prototxt_path):
            print self.model_path, "not exists, abort."
            exit()

        self.encoder_net = caffe.Net(self.deploy_path, self.model_path)

    def encode_image(self, image_path):
        img = cv2.imread(image_path)
        if img.shape[0] % 8 >0:
           img=img[0:img.shape[0]-img.shape[0]%8,:]
        if img.shape[1] % 8 >0:
           img=img[:,0:img.shape[1]-img.shape[1]%8]
        self.encoder_net.blobs['data'].reshape(1, 3, img.shape[0], img.shape[1])
        self.encoder_net.blobs['data'].data[...]=(img.transpose(2, 0, 1)-127.5)/127.5
        self.encoder_net.forward()
        net.blobs['epack'].data[i].astype(np.uint8)
        net.blobs['elabel'].data[i,0,0,0]-1
