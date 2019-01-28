import imghdr, imageio
from math import floor
import glob, cv2, os, numpy as np, sys, caffe
from utils import jigsaw
import argparse

caffe.set_mode_gpu()
caffe.set_device(0)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--deploypath',type=str,default='./prototxt/cwic_deploy.prototxt')
    parser.add_argument('--modelpath',type=str,default='./cwic_training_output_fixw0001_ADAM/snapshot-_iter_200000.caffemodel')
    parser.add_argument('--layername',type=str,default=None)
    parser.add_argument('--outputdir', type=str, default='./feature_map')

    return parser.parse_args()
print "Parsing arguments..."
args = get_arguments()
preset_list = ['conv1', 'conv2', 'conv4', 'mgdata', 'imp_conv2', 'imap', 'imap_gdata', 'inv_conv1', 'inv_conv2', 'inv_conv3', 'inv_conv4', 'pdata']
image_path = args.image_path
if not os.path.isfile(image_path):
    print image_path, "not exists, abort."
    exit()

deploy_path = args.deploypath
model_path = args.modelpath
layer_name = args.layername
outputdir = args.outputdir
outputdir= os.path.join(outputdir, os.path.basename(image_path).split('.')[0])
if not os.path.isdir(outputdir):
    os.makedirs(outputdir)

caffe.set_device(0)
caffe.set_mode_gpu()
net=caffe.Net(deploy_path,model_path,caffe.TEST)

layer_list= []
if layer_name is not None:
    layer_list.append(layer_name)
else:
    layer_list=preset_list

for layer_name in layer_list:
    image_prefix = os.path.basename(image_path).split('.')[0] + '_' + layer_name
    frame_name_wildcard = layer_name+'_channel_%s.jpg'
    cur_dir = os.path.join(outputdir, image_prefix)
    if not os.path.isdir(cur_dir):
        os.makedirs(cur_dir)

    # cur_frame = np.array(self.frames[0:self.video_length])
    # input_data = np.transpose(cur_frame[None, ...], (0, 2, 1, 3, 4))
    # self.video_net.blobs['data'].data[...] = input_data
    # self.video_net.forward()
    # feature_map = self.video_net.blobs[layer_name].data[0, ...]
    img = cv2.imread(image_path)
    shape = (img.shape[1], img.shape[0])
    print shape
    if img.shape[0] % 8 >0:
       img=img[0:img.shape[0]-img.shape[0]%8,:]
    if img.shape[1] % 8 >0:
       img=img[:,0:img.shape[1]-img.shape[1]%8]  
    net.blobs['data'].reshape(1,3,img.shape[0],img.shape[1])
    data=(img.transpose(2,0,1)-127.5)/127.5
    net.blobs['data'].data[...] = data
    net.forward()

    feature_maps = net.blobs[layer_name].data[...][0]
    print layer_name, "Feature shape:", feature_maps.shape
    # print np.min(feature_maps), np.max(feature_maps)
    channel_num = len(feature_maps)
    for channel in range(channel_num):
        frame = feature_maps[channel]
        # print "Frame shape:", frame.shape
        channel_index = str(channel)
        # cv2.imwrite(frame_path, frame) 
        # frame = frame - np.min(frame)
        # frame = frame / np.max(frame)
        # frame = frame * 255
        frame = frame*127.5 + 127.5
        frame = cv2.resize(frame, dsize = shape)
        # print np.min(frame), np.max(frame)
        # print frame.shape;#exit()
        frame_name = frame_name_wildcard % str(channel+1).zfill(3)
        frame_path = os.path.join(cur_dir, frame_name)
        
        cv2.imwrite(frame_path, frame) 
        # # for i in range(len(frames)):
        # for i in range(1):
        #     frame = frames[i]
        #     # print np.min(frame), np.max(frame),

    outputpath = os.path.join(outputdir, image_prefix + '.jpg')

    jigsaw(imageDir=cur_dir, output_path=outputpath, stdsize=shape, padding=2)    

