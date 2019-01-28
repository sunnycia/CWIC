import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_weights(net, layer_name, padding=0, filename='', visualize=False):
    # refer to 
    #     https://www.eriksmistad.no/visualizing-learned-features-of-a-caffe-neural-network/
    
    # The parameters are a list of [weights, biases]
    data = np.copy(net.params[layer_name][0].data)
    # N is the total number of convolutions
    N = data.shape[0]*data.shape[1]
    # Ensure the resulting image is square
    filters_per_row = int(np.ceil(np.sqrt(N)))
    # Assume the filters are square
    filter_size = data.shape[2]
    # Size of the result image including padding
    result_size = filters_per_row*(filter_size + padding) - padding
    # Initialize result image to all zeros
    result = np.zeros((result_size, result_size))

    # Tile the filters into the result image
    filter_x = 0
    filter_y = 0
    for n in range(data.shape[0]):
        for c in range(data.shape[1]):
            if filter_x == filters_per_row:
                filter_y += 1
                filter_x = 0
            for i in range(filter_size):
                for j in range(filter_size):
                    result[filter_y*(filter_size + padding) + i, filter_x*(filter_size + padding) + j] = data[n, c, i, j]
            filter_x += 1

    # Normalize image to 0-1
    min = result.min()
    max = result.max()
    result = (result - min) / (max - min)
    # print result.shape;exit()
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(result, cmap='gray', interpolation='nearest')
    # Plot figure
    if visualize:
        plt.show()
    else:
        result = result * 255
        cv2.imwrite(filename, result)
        # Save plot if filename is set
        # if filename != '':
        #     plt.savefig(filename)
            # plt.savefig(filename, bbox_inches='tight', pad_inches=0)

def visualize_3dweights(net, layer_name, padding=0, filename='', visualize=False):
    # refer to 
    #     https://www.eriksmistad.no/visualizing-learned-features-of-a-caffe-neural-network/
    
    # The parameters are a list of [weights, biases]
    data = np.copy(net.params[layer_name][0].data)
    # N is the total number of convolutions
    N = data.shape[0]*data.shape[1]
    # Ensure the resulting image is square
    filters_per_row = int(np.ceil(np.sqrt(N)))
    # Assume the filters are square
    assert data.shape[3]==data.shape[4]
    filter_size = data.shape[3]
    # Size of the result image including padding
    result_size = filters_per_row*(filter_size + padding) - padding
    # Initialize result image to all zeros
    result = np.zeros((result_size, result_size, data.shape[2]))

    # Tile the filters into the result image
    filter_x = 0
    filter_y = 0
    # filter_z = 0
    for n in range(data.shape[0]):
        for c in range(data.shape[1]):
            if filter_x == filters_per_row:
                filter_y += 1
                filter_x = 0
            for d in range(data.shape[2]):
                for i in range(filter_size):
                    for j in range(filter_size):
                        result[filter_y*(filter_size + padding) + i, filter_x*(filter_size + padding) + j, d] = data[n, c, d, i, j]
            filter_x += 1

    # Normalize image to 0-1
    min = result.min()
    max = result.max()
    result = (result - min) / (max - min)
    # print result.shape;exit()
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(result, cmap='gray', interpolation='nearest')
    # Plot figure
    if visualize:
        plt.show()
    else:
        result = result * 255
        cv2.imwrite(filename, result)
        # Save plot if filename is set
        # if filename != '':
        #     plt.savefig(filename)
            # plt.savefig(filename, bbox_inches='tight', pad_inches=0)

class SimpleTransformer:

    """
    SimpleTransformer is a simple class for preprocessing and deprocessing
    images for caffe.
    """

    def __init__(self, mean=[128, 128, 128]):
        self.mean = np.array(mean, dtype=np.float32)
        self.scale = 1.0

    def set_mean(self, mean):
        """
        Set the mean to subtract for centering the data.
        """
        self.mean = mean

    def set_scale(self, scale):
        """
        Set the data scaling.
        """
        self.scale = scale

    def preprocess(self, im):
        """
        preprocess() emulate the pre-processing occurring in the vgg16 caffe
        prototxt.
        """

        im = np.float32(im)
        im = im[:, :, ::-1]  # change to BGR
        im -= self.mean
        im *= self.scale
        im = im.transpose((2, 0, 1))

        return im

    def deprocess(self, im):
        """
        inverse of preprocess()
        """
        im = im.transpose(1, 2, 0)
        im /= self.scale
        im += self.mean
        im = im[:, :, ::-1]  # change to RGB

        return np.uint8(im)

class CaffeNetwork:
    # def __init__(self, )
    pass

class CaffeSolver:

    """
    Caffesolver is a class for creating a solver.prototxt file. It sets default
    values and can export a solver parameter file.
    Note that all parameters are stored as strings. Strings variables are
    stored as strings in strings.
    """
    def __init__(self, trainnet_prototxt_path="train.prototxt"):

        self.sp = {}

        # critical:
        self.sp['base_lr'] = '0.01'
        self.sp['momentum'] = '0.9'

        # looks:
        self.sp['display'] = '100'
        self.sp['iter_size'] = '1'

        # learning rate policy
        self.sp['lr_policy'] = '"inv"'

        # important, but rare:
        self.sp['gamma'] = '0.0001'
        self.sp['power'] = '0.75'
        self.sp['weight_decay'] = '0.0005'
        self.sp['train_net'] = '"' + trainnet_prototxt_path + '"'

        #
        self.sp['solver_mode'] = 'GPU'
        self.sp['solver_type'] = 'ADADELTA'
        self.sp['delta'] = '1e-6'

        #snapshot
        self.sp['snapshot'] = '50000'
        self.sp['snapshot_prefix'] = '"../training_output/salicon/snapshot"'

    def update_solver(self, update_dict):
        for key in update_dict:
            self.sp[key] = update_dict[key]


    def add_from_file(self, filepath):
        """
        Reads a caffe solver prototxt file and updates the Caffesolver
        instance parameters.
        """
        with open(filepath, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                splitLine = line.split(':')
                self.sp[splitLine[0].strip()] = splitLine[1].strip()

    def write(self, filepath):
        """
        Export solver parameters to INPUT "filepath". Sorted alphabetically.
        """
        f = open(filepath, 'w')
        for key, value in sorted(self.sp.items()):
            if not(type(value) is str):
                raise TypeError('All solver parameters must be strings')
            f.write('%s: %s\n' % (key, value))


