import os
import caffe
import argparse
from caffe_tools import visualize_weights, visualize_3dweights

# python visualize_weight.py --deploypath='prototxt/deploy_3layer_deconv.prototxt' 
# --model='../training_output/salicon/train_kldloss_withouteuc-batch-8_1509584263/snapshot-_iter_100000.caffemodel' 
# --output='upsample1.jpg' --layer='upsample_1'

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str, required=True)
    parser.add_argument('--deploypath', type=str, required=True)
    parser.add_argument('--outputpath', type=str, required=True)
    parser.add_argument('--viz', type=bool, default=False)
    parser.add_argument('--type', type=str, default='2d', help='2d or 3d')

    parser.add_argument('--layer', type=str, default='conv1')
    return parser.parse_args()

print "Parsing argument.."
args = get_arguments()

net = caffe.Net(args.deploypath, args.modelpath, caffe.TEST)

##check if output directory exists
output_path = args.outputpath
output_dir = os.path.dirname(args.outputpath)
if not output_dir == '':
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

if not args.layer == 'all':
    if args.type=='2d':
        visualize_weights(net, args.layer, filename=output_path, visualize=args.viz, padding=1)
    elif args.type=='3d':
        visualize_3dweights(net, args.layer, filename=output_path, visualize=args.viz, padding=1)


else:
    #NOT IMPLEMENT YET
    pass