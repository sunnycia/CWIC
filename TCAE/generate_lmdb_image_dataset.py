import os
import cv2
import numpy as np
import lmdb
import caffe
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, required=True)
parser.add_argument('--width', type=int, default=128)
parser.add_argument('--height', type=int, default=128)
parser.add_argument('--lmdb_dir', type=str, default='mylmdb')

args = parser.parse_args()
image_dir = args.image_dir 
lmdb_dir = args.lmdb_dir 
if os.path.isdir(lmdb_dir):
    choice=raw_input("%s already exists, do you want to remove it?(y/n):" % lmdb_dir)
    if choice=='y' or choice=='Y':
        shutil.rmtree(lmdb_dir)
    else:
        print "Change --lmdb_dir."
        exit()
width = args.width 
height = args.height 
step = int(width/2)

image_path_list = os.listdir(image_dir)
crop_image_list = []
for image_path in image_path_list:
    image = cv2.imread(os.path.join(image_dir, image_path))
    img_height, img_width, img_channel = image.shape
    for h in range(0, img_height, step):
        if (h+height) > img_height:
            break
        for w in range(0, img_width, step):
            if (w+width) > img_width:
                break # crop out of boundary
            X = image[h:h+height, w:w+width, ...]
            assert X.shape==(height, width, 3)
            crop_image_list.append(X)

N = len(crop_image_list)

# Let's pretend this is interesting data
X = np.zeros((N, 3, height, width), dtype=np.uint8)

# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.
map_size = X.nbytes * 1.5
print "%d samples, require %s map size" % (N, str(map_size)) 
env = lmdb.open(lmdb_dir, map_size=map_size)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        
        crop_image = np.transpose(crop_image_list[i], (2, 1, 0))
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = crop_image.shape[0]
        datum.height = crop_image.shape[1]
        datum.width = crop_image.shape[2]
        datum.data = crop_image.tobytes()
        str_id = '{:08}'.format(i)
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
    
print "Done."