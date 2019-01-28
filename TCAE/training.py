import sys
import caffe
import numpy as np
if __name__ == '__main__':
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver=caffe.AdamSolver('./prototxt/adam_solver.prototxt')
    # solver.restore('./save/cmp_iter_10000.solverstate')
    #solver.net.copy_from('./save/cmp_iter_60000.caffemodel')
    while True:
        solver.step(1)
        print solver.net.blobs['data'].data.max(), solver.net.blobs['data'].data.min(), solver.net.blobs['data'].data.mean(),
        print 'loss:', solver.net.blobs['loss'].data, '\r',
        sys.stdout.flush()
    # for i in range(10000):
    #     solver.step(10)