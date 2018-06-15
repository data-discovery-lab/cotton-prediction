import numpy as np
import os
# Use second GPU -- change if you want to use a first one
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Dataset:
    def __init__(self, train, test, val_frac=0.5, shuffle=True, scale_func=None):
        split_idx = int(len(test['y']) * (1 - val_frac))
        self.test_x, self.valid_x = test['X'][:, :, :, :split_idx], test['X'][:, :, :, split_idx:]
        self.test_y, self.valid_y = test['y'][:split_idx], test['y'][split_idx:]
        self.train_x, self.train_y = train['X'], train['y']
        # The SVHN dataset comes with lots of labels, but for the purpose of this exercise,
        # we will pretend that there are only 1000.
        # We use this mask to say which labels we will allow ourselves to use.
        self.label_mask = np.zeros_like(self.train_y)
        self.label_mask[0:1000] = 1

        # Roll the specified axis backwards, until it lies in a given position.
        # From (32, 32, 3, 73257) to (73257, 32, 32, 3)
        self.train_x = np.rollaxis(self.train_x, axis=3)
        self.valid_x = np.rollaxis(self.valid_x, axis=3)
        self.test_x = np.rollaxis(self.test_x, axis=3)

        if scale_func is None:
            self.scaler = self.scale
        else:
            self.scaler = scale_func
        self.train_x = self.scaler(self.train_x)
        self.valid_x = self.scaler(self.valid_x)
        self.test_x = self.scaler(self.test_x)
        self.shuffle = shuffle

        # trainset = np.rollaxis(train['X'], 3)
        # testset = np.rollaxis(test['X'], 3)
        #
        # self.dataset = np.concatenate((trainset, testset), axis=0)

    def scale(self, x, feature_range=(-1, 1)):
        # scale to (0, 1)
        x = ((x - x.min()) / (255 - x.min()))

        # scale to feature_range
        min, max = feature_range
        x = x * (max - min) + min
        return x


    def batches(self, batch_size, which_set="train"):
        x_name = which_set + "_x"
        y_name = which_set + "_y"

        # Return the value of the named attribute of object
        num_examples = len(getattr(self, y_name))
        if self.shuffle:
            idx = np.arange(num_examples)
            np.random.shuffle(idx)
            setattr(self, x_name, getattr(self, x_name)[idx])
            setattr(self, y_name, getattr(self, y_name)[idx])
            if which_set == "train":
                self.label_mask = self.label_mask[idx]

        dataset_x = getattr(self, x_name)
        dataset_y = getattr(self, y_name)
        for ii in range(0, num_examples, batch_size):
            x = dataset_x[ii:ii + batch_size]
            y = dataset_y[ii:ii + batch_size]

            if which_set == "train":
                # When we use the data for training, we need to include
                # the label mask, so we can pretend we don't have access
                # to some of the labels, as an exercise of our semi-supervised
                # learning ability
                # x: [BATCH_SIZE, 32, 32, 3]
                # y: [BATCH_SIZE, 1]
                # label_mask: [BATCH_SIZE, 1] whether a 0: Label Cannot be used or 1: Label can be used
                yield x, y, self.label_mask[ii:ii + batch_size]
            else:
                yield x, y