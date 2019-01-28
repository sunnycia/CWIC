#coding=utf-8
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2 as P2

def training_solver(net_path='train.prototxt', snapshot_prefix='training_output/snapshot-'):
    root_str='./'
    my_project_root = root_str#my-caffe-project目录
    sovler_string = caffe.proto.caffe_pb2.SolverParameter() 

    sovler_string.train_net = net_path
    sovler_string.snapshot_prefix = snapshot_prefix
    sovler_string.base_lr = 0.0001
    sovler_string.momentum = 0.9 
    sovler_string.power = 0.75 
    sovler_string.gamma = 0.0001
    sovler_string.weight_decay = 5*1e-4
    # inv':return base_lr * (1 + gamma * iter) ^ (- power)
    sovler_string.lr_policy = 'inv' 
    sovler_string.solver_type = P2.SolverParameter.ADADELTA 
    sovler_string.display = 100
    # sovler_string.max_iter = 200000
    sovler_string.snapshot = 10000 

    #sovler_string.snapshot_format = 0 

    sovler_string.solver_mode = P2.SolverParameter.GPU 

    return sovler_string
if __name__=='__main__':
    solver = training_solver()
    print str(solver)