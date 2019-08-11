from torch.utils.data import Dataset
from torch.utils.data import Subset
from PIL import Image
import cv2
import numpy as np
import csv

class CustomDataset(Dataset):
    '''
    Loading custom dataset into standard pytorch dataset structure
    -- assumes dataset in 'root' directory
    -- transform: applies this transformation to the dataset before feeding to the neural net (dependent on how the
    --  original neural net was trained)
    -- labeled_thresh: any classes with less than 'labeled_thresh' number of examples present in the dataset will be
    --  left out of training, as having too little training data from a class will result in bad predictions.

    '''

    def __init__(self, root, transform=None, labeled_thresh=None):
        self.root = root
        self.transform = transform
        self.labeled_thresh = labeled_thresh
        self.grab_data()

    def grab_data(self):
        print("Processing CustomDataset dataset...")

        # read csv file with image names and labels
        with open(self.root + '/img-labels.csv') as labels_csv:
            reader = csv.reader(labels_csv, delimiter=',')
            img_labels = [img for img in reader]

        # extract all labels and create class list
        full_labels = []
        for img in img_labels:
            full_labels.extend(img[1:])
        self.class_list = sorted(list(set(full_labels)))

        # create and populate label vectors
        self.labels = np.zeros([len(img_labels), len(self.class_list)], dtype=np.float32)
        for i, img in enumerate(img_labels):
            for l in img[1:]:
                self.labels[i][self.class_list.index(l)] = 1

        # delete the labels that have less labeled examples than the threshold allows
        delete_ind = []
        delete_name = []
        if self.labeled_thresh is not None:
            for i, col in enumerate(self.labels.sum(0)):
                if col < self.labeled_thresh:
                    delete_ind.append(i)
                    delete_name.append(self.class_list[i])
            self.labels = np.delete(self.labels, delete_ind, 1)
            for i in sorted(delete_ind, reverse=True):
                del self.class_list[i]

        # load image data (unless their labels have all been deleted)
        self.data = []
        self.file_names = []
        delete_ind = []
        for i, img in enumerate(img_labels):
            if self.labels[i].sum() != 0:
                d = cv2.imread(self.root + '/' + img[0])
                self.data.append(d)
                self.file_names.append(self.root + '/' + img[0])
            else:
                delete_ind.append(i)
        self.labels = np.delete(self.labels, delete_ind, 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]
        # data = data.transpose((2, 0, 1))  # convert to HWC

        # returning data as PIL image is consistent with other PyTorch datasets
        data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, label

    # Description: this function will create a pytorch random Subset of the dataset
    # Input:
    #   per_class: number of examples per class to sample from dataset
    #       -- if set to 'min' or an invalid number, value will be the minimum possible, as determined by the dataset
    def split_dataset(self, per_class='min'):
        num_classes = len(self.class_list)
        class_counts = np.sum(self.labels, 0)

        if per_class == 'min':
            per_class = min(class_counts)
        elif per_class > min(class_counts):
            print("ERROR: not enough examples for class " + str(class_counts.index(min(class_counts))))
            print("Setting per_class to min(class_counts)...")
            per_class = min(class_counts)

        # randomize order of data
        perm = np.random.permutation(len(self.data))

        # define and create new dataset
        subset_inds = []

        # we can use the same image for multiple classes, but we don't want duplicate images. flags will make sure we
        #   dont duplicate images in the dataset.
        flags = np.zeros(len(self.data))
        for c in range(num_classes):
            iter = 0
            for p in range(int(per_class)):
                while True:
                    if self.labels[perm[iter]][c] == 1:  # if current image has the class we're looking for
                        if flags[perm[iter]] == 0:  # if the current image isn't in our new dataset already
                            subset_inds.append(perm[iter])
                            flags[perm[iter]] = 1
                        iter += 1
                        break
                    iter += 1

        # shuffle X_new and Y_new in same order
        subset_inds = np.random.permutation(subset_inds)

        # create subset
        return Subset(self, subset_inds)