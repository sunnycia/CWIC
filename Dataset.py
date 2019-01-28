import os, glob, cv2, numpy as np
import random
import sys

class ImageDataset():
    def __init__(self, image_dir, training_size=(128, 128), training_example_props=0.8):
        self.training_size = training_size
        imagepath_list = glob.glob(os.path.join(image_dir, '*.*'))
        self.data = []
        for imagepath in imagepath_list:
            print len(self.data), 'examples in total', '\r', 
            sys.stdout.flush()
            # self.data.append(self.pre_process_image(cv2.imread(imagepath, 1)))
            # self.data.append(imagepath)
            self.data.append(cv2.imread(imagepath, 1))


        # random.shuffle(self.data)

        self.examples = len(self.data)

        self.completed_epoch = 0
        self.index_in_epoch = 0
        print self.examples, "in total."
    def pre_process_image(self, image):
        ## image is in BGR channel order
        return (image-127.5)/127.5

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self.index_in_epoch
        self.index_in_epoch += batch_size

        if self.index_in_epoch > self.examples:
            # Finished epoch
            # print "Finished epoch"
            self.completed_epoch += 1
            # Shuffle the data
            random.shuffle(self.data)

            # perm = np.arange(self.examples)
            # np.random.shuffle(perm)
            # self.data = self.data[perm]
            # self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.examples
        end = self.index_in_epoch

        mini_batch = []
        # print start, 'to',  end
        for i in range(start, end):
            # img = cv2.imread(self.data[i], 1)
            img = self.data[i]
            # print i, 
            # print self.data[i].shape
            width = img.shape[1]
            height = img.shape[0]

            upper_left_pos = (random.randint(0, width-self.training_size[0]), random.randint(0, height-self.training_size[1]))
            

            sub_image = img[upper_left_pos[1]:upper_left_pos[1]+self.training_size[1], upper_left_pos[0]:upper_left_pos[0]+self.training_size[0]]
            print sub_image.shape, '\r', 
            sys.stdout.flush()
            assert sub_image.shape == (self.training_size[0], self.training_size[1], 3)
 
            mini_batch.append(np.transpose(self.pre_process_image(sub_image), (2, 0, 1))) ## channel, height, width

        return mini_batch


## FOR TEST
if __name__ == '__main__':
    dataset = ImageDataset('/data/sunnycia/image_compression_challenge/dataset/CLIC_PRO', training_size=(128, 128), training_example_props=0.8)

    for i in range(100000):
        hey = dataset.next_batch(8)