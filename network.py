#coding=utf-8
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2 as P2
def cwic_network(batch_size=16, img_size=(128, 128, 3), output_feat_channel=64, compression_ratio=0.18, pen_weight=0.03, groups=16):
    n = caffe.NetSpec()
    input_shape_list = [str(batch_size), str(img_size[2]), str(img_size[0]), str(img_size[1])]
    input_shape_str = ','.join(input_shape_list)

    n.data = L.Python(python_param=dict(module='CustomData', 
                                        layer='CustomData',
                                        param_str=input_shape_str))
    
    # print P.Dtow.DtowMethod.DESCRIPTOR.values_by_name['MDTOW'].number;exit()
    # print caffe.proto.caffe_pb2.DtowParameter.MDTOW;exit()

    ### ╔═╗╔╗╔╔═╗╔═╗╔╦╗╔═╗╦═╗
    ### ║╣ ║║║║  ║ ║ ║║║╣ ╠╦╝
    ### ╚═╝╝╚╝╚═╝╚═╝═╩╝╚═╝╩╚═
    ##Convolution 1
    n.conv1 = L.Convolution(bottom='data', num_output=128, kernel_size=8, stride=4, pad=2, weight_filler=dict(type='xavier'))
    n.conv1_relu = L.ReLU(bottom='conv1', in_place=True, top='conv1')

    ##Block 1
    n.blk1_branch2b = L.Convolution(bottom='conv1', num_output=128, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'))
    n.blk1_branch2b_relu = L.ReLU(bottom='blk1_branch2b', in_place=True, top='blk1_branch2b')
    n.blk1_branch2c = L.Convolution(bottom='blk1_branch2b', num_output=128, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'))
    n.blk1_branch2c_relu = L.ReLU(bottom='blk1_branch2c', in_place=True, top='blk1_branch2c')
    n.blk1 = L.Eltwise(bottom=['conv1','blk1_branch2c'])
    n.blk1_relu = L.ReLU(bottom='blk1', in_place=True, top='blk1')

    ##Convolution 2
    n.conv2 = L.Convolution(bottom='blk1', num_output=256, kernel_size=4, stride=2, pad=1, weight_filler=dict(type='xavier'))
    n.conv2_relu = L.ReLU(bottom='conv2', in_place=True, top='conv2')

    ##Block 2
    n.blk2_branch2b = L.Convolution(bottom='conv2', num_output=128, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'))
    n.blk2_branch2b_relu = L.ReLU(bottom='blk2_branch2b', in_place=True, top='blk2_branch2b')
    n.blk2_branch2c = L.Convolution(bottom='blk2_branch2b', num_output=256, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'))
    n.blk2_branch2c_relu = L.ReLU(bottom='blk2_branch2c', in_place=True, top='blk2_branch2c')
    n.blk2 = L.Eltwise(bottom=['conv2', 'blk2_branch2c'])
    n.blk2_relu = L.ReLU(bottom='blk2', in_place=True, top='blk2')

    ##Convolution 3
    n.conv3 = L.Convolution(bottom='blk2', num_output=256, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'))
    n.conv3_relu = L.ReLU(bottom='conv3', in_place=True, top='conv3')

    ##Block 3
    n.blk3_branch2b = L.Convolution(bottom='conv3', num_output=128, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'))
    n.blk3_branch2b_relu = L.ReLU(bottom='blk3_branch2b', in_place=True, top='blk3_branch2b')
    n.blk3_branch2c = L.Convolution(bottom='blk3_branch2b', num_output=256, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'))
    n.blk3_branch2c_relu = L.ReLU(bottom='blk3_branch2c', in_place=True, top='blk3_branch2c')
    n.blk3 = L.Eltwise(bottom=['conv3', 'blk3_branch2c'])
    n.blk3_relu = L.ReLU(bottom='blk3', in_place=True, top='blk3')

    ## Convolution 4
    n.conv4 = L.Convolution(bottom='blk3', num_output=output_feat_channel, kernel_size=1, stride=1, pad=0, weight_filler=dict(type='xavier'))
    n.conv4_sig = L.Sigmoid(bottom='conv4', in_place=True, top='conv4')

    ## Feature Binarizer
    n.mgdata = L.Round(bottom='conv4', round_param=dict(scale=0.01))

    ## Importance map network
    n.imp_conv1 = L.Convolution(bottom='blk3', num_output=128, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'))
    n.imp_conv1_relu = L.ReLU(bottom='imp_conv1', in_place=True, top='imp_conv1')
    n.imp_conv2 = L.Convolution(bottom='imp_conv1', num_output=1, kernel_size=1, stride=1, pad=0, weight_filler=dict(type='xavier'))
    n.imp_conv2_sig = L.Sigmoid(bottom='imp_conv2', in_place=True, top='imp_conv2')

    # enum ImpMethod{
    #     GLOBAL = 0;
    #     LOCAL = 1;
    # }
    n.imap = L.ImpMap(bottom='imp_conv2', imp_map_param=dict(method=P.ImpMap.ImpMethod.DESCRIPTOR.values_by_name['GLOBAL'].number, 
                                                            lquantize=False, 
                                                            groups=groups, 
                                                            cmp_ratio=compression_ratio, 
                                                            weight=pen_weight, 
                                                            channel_out=output_feat_channel))

    n.imap_gdata = L.Eltwise(bottom=['imap', 'mgdata'], eltwise_param=dict(operation=P.Eltwise.EltwiseOp.DESCRIPTOR.values_by_name['PROD'].number))


    ### ╔╦╗╔═╗╔═╗╔═╗╔╦╗╔═╗╦═╗
    ###  ║║║╣ ║  ║ ║ ║║║╣ ╠╦╝
    ### ═╩╝╚═╝╚═╝╚═╝═╩╝╚═╝╩╚═

    ## Convolution 5
    n.inv_conv1 = L.Convolution(bottom='imap_gdata', num_output=512, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'))
    n.inv_conv1_relu = L.ReLU(bottom='inv_conv1', in_place=True, top='inv_conv1')
    
    ## Block 4
    n.blk4_branch2b = L.Convolution(bottom='inv_conv1', num_output=128, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'))
    n.blk4_branch2b_relu = L.ReLU(bottom='blk4_branch2b', in_place=True, top='blk4_branch2b')
    n.blk4_branch2c = L.Convolution(bottom='blk4_branch2b', num_output=512, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'))
    n.blk4_branch2c_relu = L.ReLU(bottom='blk4_branch2c', in_place=True, top='blk4_branch2c')
    n.blk4 = L.Eltwise(bottom=['inv_conv1', 'blk4_branch2c'])
    n.blk4_relu = L.ReLU(bottom='blk4', in_place=True, top='blk4')

    ## Convolution 6
    n.inv_conv2 = L.Convolution(bottom='blk4', num_output=512, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'))
    n.inv_conv2_relu = L.ReLU(bottom='inv_conv2', in_place=True, top='inv_conv2')

    ##Block 5
    n.blk5_branch2b = L.Convolution(bottom='inv_conv2', num_output=128, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'))
    n.blk5_branch2b_relu = L.ReLU(bottom='blk5_branch2b', in_place=True, top='blk5_branch2b')
    n.blk5_branch2c = L.Convolution(bottom='blk5_branch2b', num_output=512, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'))
    n.blk5_branch2c_relu = L.ReLU(bottom='blk5_branch2c', in_place=True, top='blk5_branch2c')
    n.blk5 = L.Eltwise(bottom=['inv_conv2', 'blk5_branch2c'])
    n.blk5_relu = L.ReLU(bottom='blk5', in_place=True, top='blk5')
    
    ## Depth to space 1
    n.dtow1 = L.Dtow(bottom='blk5', dtow_param=dict(psize=2))

    ## Convolution 7
    n.inv_conv3 = L.Convolution(bottom='dtow1', num_output=256, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'))
    n.inv_conv3_relu = L.ReLU(bottom='inv_conv3', in_place=True, top='inv_conv3')

    ##Block 6
    n.blk6_branch2b = L.Convolution(bottom='inv_conv3', num_output=128, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'))
    n.blk6_branch2b_relu = L.ReLU(bottom='blk6_branch2b', in_place=True, top='blk6_branch2b')
    n.blk6_branch2c = L.Convolution(bottom='blk6_branch2b', num_output=256, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'))
    n.blk6_branch2c_relu = L.ReLU(bottom='blk6_branch2c', in_place=True, top='blk6_branch2c')
    n.blk6 = L.Eltwise(bottom=['inv_conv3', 'blk6_branch2c'])
    n.blk6_relu = L.ReLU(bottom='blk6', in_place=True, top='blk6')

    ## Depth to space 2
    n.dtow2 = L.Dtow(bottom='blk6', dtow_param=dict(psize=4))

    ## Convolution 8
    n.inv_conv4 = L.Convolution(bottom='dtow2', num_output=32, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'))
    n.inv_conv4_relu = L.ReLU(bottom='inv_conv4', in_place=True, top='inv_conv4')

    ## Predict
    n.pdata = L.Convolution(bottom='inv_conv4', num_output=3, kernel_size=1, stride=1, pad=0, weight_filler=dict(type='xavier'))

    n.loss = L.EuclideanLoss(bottom=['pdata', 'data'])

    return n.to_proto()
if __name__=='__main__':
    network_proto = cwic_network()
    with open('atest.prototxt', 'w') as f:
        f.write(str(n.to_proto()))
    exit()