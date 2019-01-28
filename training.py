from Dataset import ImageDataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse, cv2, os, glob, sys, time
import cPickle as pkl
import numpy as np
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2 as P2
from random import shuffle
import google.protobuf.text_format as txtf
from network import cwic_network
from solver import training_solver

from utils import *

caffe.set_mode_gpu()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_dir', type=str, required=True, help='training data directory')
    parser.add_argument('--valid_data_dir', type=str, required=True, help='training data directory')
    parser.add_argument('--solver_prototxt', type=str, default='prototxt/solver.prototxt', help='the network prototxt')
    parser.add_argument('--pretrained_model', type=str, default='model/ResNet-50-model.caffemodel', help='Snapshot path.')

    parser.add_argument('--use_snapshot', type=str, default=None, help='Snapshot path.')
    parser.add_argument('--trainingexampleprops', type=float, default=0.8, help="")

    parser.add_argument('--model_name', type=str, default='model', help='Extra model information.')
    parser.add_argument('--extrainfo_str', type=str, default='', help='Extra model information.')
    
    #Network parameter
    parser.add_argument('--batch', type=int, default=16, help='training mini batch')
    parser.add_argument('--feat_channel', type=int, default=64)
    parser.add_argument('--groups', type=int, default=16)
    parser.add_argument('--compression_ratio', type=float, default=1.0)
    parser.add_argument('--pen_weight', type=float, default=0.03)


    #default learning parameter
    parser.add_argument('--train_prototxt', type=str,   default='prototxt/resnet_autoencoder.prototxt', help='Network arch prototxt')
    parser.add_argument('--snapshot_dir',   type=str,   default='training_output', help='Snapshot saving directory.')
    parser.add_argument('--lr_policy',      type=str,   default='fixed')
    parser.add_argument('--stepsize',       type=int,   default=10000)
    parser.add_argument('--base_lr',        type=float, default=0.0001)
    parser.add_argument('--gamma',          type=float, default=0.0001)
    parser.add_argument('--momentum',       type=float, default=0.9)
    
    parser.add_argument('--power',          type=float, default=0.75)
    parser.add_argument('--weight_decay',   type=float, default=5*1e-4)
    
    parser.add_argument('--display',        type=int,   default=100)
    parser.add_argument('--max_iter',       type=int,   default=200000)
    parser.add_argument('--snapshot',       type=int,   default=10000)
    parser.add_argument('--solver_mode',    type=str,   default='GPU')
    parser.add_argument('--solver_type',    type=str,   default='ADAM')
    return parser.parse_args()

print "Parsing arguments..."
args = get_arguments()
mini_batch = args.batch
training_data_dir = args.training_data_dir
train_prototxt_path = args.train_prototxt
solver_prototxt_path = args.solver_prototxt
snapshot_dir= args.snapshot_dir
if not os.path.isdir(snapshot_dir):
    os.makedirs(snapshot_dir)
snapshot_prefix = os.path.join(snapshot_dir, 'snapshot-')
plot_figure_dir = os.path.join(snapshot_dir, 'figure')
if not os.path.isdir(plot_figure_dir):
    os.makedirs(plot_figure_dir)
print "Loss figure will be save to", plot_figure_dir

## Modify Network here
net_proto = cwic_network(batch_size=mini_batch, 
                        output_feat_channel=args.feat_channel, 
                        compression_ratio=args.compression_ratio, 
                        pen_weight=args.pen_weight,
                        groups=args.groups)

with open(train_prototxt_path, 'w') as f:
    f.write(str(net_proto));#exit()
##

## Modify learning parameter here
solver = caffe.proto.caffe_pb2.SolverParameter() 
solver.train_net=train_prototxt_path
solver.snapshot_prefix=snapshot_prefix
solver.base_lr = args.base_lr
# solver.gamma = args.gamma
solver.momentum = args.momentum

# solver.power = args.power
# solver.weight_decay = args.weight_decay

solver.display = args.display
solver.max_iter = args.max_iter
solver.snapshot = args.snapshot
solver.lr_policy = args.lr_policy
solver.solver_mode = P.Solver.SolverMode.DESCRIPTOR.values_by_name[args.solver_mode].number
# solver.solver_type = P.Solver.SolverType.DESCRIPTOR.values_by_name[args.solver_type].number
with open(solver_prototxt_path, 'w') as f:
    f.write(str(solver))
##


## Copy network and  solver to snapshot directory
with open(os.path.join(snapshot_dir, 'network.prototxt'), 'w') as f:
    f.write(str(net_proto))
with open(os.path.join(snapshot_dir, 'solver.prototxt'), 'w') as f:
    f.write(str(solver))
##



pretrained_model_path= args.pretrained_model
snapshot_path = args.use_snapshot
#Check if snapshot exists
if snapshot_path is not None:
    if not os.path.isfile(snapshot_path):
        print snapshot_path, "not exists.Abort"
        exit()
solver=''
if args.solver_type=='SGD':
    solver = caffe.SGDSolver(solver_prototxt_path)
if args.solver_type=='ADADELTA':
    solver = caffe.ADADELTASolver(solver_prototxt_path)
if args.solver_type=='ADAM':
    solver = caffe.AdamSolver(solver_prototxt_path)

if args.use_snapshot == None:
    if pretrained_model_path != '':
        solver.net.copy_from(pretrained_model_path) # untrained.caffemodel
else:
    solver.restore(snapshot_path)

tranining_dataset = ImageDataset(image_dir=training_data_dir)

tart_time = time.time()

max_iter = 50000000
validation_iter = 1000
plot_iter = 500
epoch=10
idx_counter = 0

x=[]
y=[]
z=[] # validation

plt.plot(x, y)
_step=0
if args.use_snapshot != None:
    _step = int(os.path.basename(args.use_snapshot).split('.')[0].split('_')[-1])

print "Start training..."
while _step * mini_batch < max_iter:

    frame_minibatch = tranining_dataset.next_batch(mini_batch)
    # print frame_minibatch.shape;exit()
    solver.net.blobs['data'].data[...] = frame_minibatch
    solver.step(1)

    ##For debug
    # print "Step forward.."
    # solver.net.forward()
    # print "Step backward.."
    # solver.net.backward()


    if _step%plot_iter==0:
        x.append(_step)
        y.append(solver.net.blobs['loss'].data[...].tolist())
        plt.plot(x, y)
        plt.xlabel('Iter')
        plt.ylabel('kld loss')
        plt.savefig(os.path.join(plot_figure_dir, "plot"+str(_step)+".png"))
        plt.clf()

        pkl.dump(x, open(os.path.join(plot_figure_dir, "x.pkl"), 'wb'))
        pkl.dump(y, open(os.path.join(plot_figure_dir, "y.pkl"), 'wb'))


    if _step%validation_iter==0:
        # do validation for validation set, and plot average 
        # metric(cc, sim, auc, kld, nss) performance dictionary
        pass
        # metric_dict = MetricValidation()


    _step+=1
