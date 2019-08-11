import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np
import random

class CustomDataset(Dataset):
    '''
    Loading custom dataset into a standard pytorch dataset structure
    -- assumes dataset in 'root' directory
    -- pick_classes: The user can choose the number of classes taken from the dataset. This argument can be an integer
    --  or a explicit list of classes (as strings).

    '''

    def __init__(self, root, transform=None, pick_classes=None):
        self.root = root
        self.class_list = os.listdir(self.root)
        if '.DS_Store' in self.class_list:
            self.class_list.remove('.DS_Store')
        self.transform = transform
        self.pick_classes = pick_classes
        self.data = []
        self.labels = []
        self.grab_data()

    def grab_data(self):
        print("Processing CustomDataset dataset...")

        # Check if pick_classes is valid. If it is and it is a valid int, set class_list to random subset of classes
        #   where len(set(class_list)) == pick_classes. If it is a valid list of strings, set class_list = to
        #   pick_classes.
        if self.pick_classes is not None:
            if type(self.pick_classes) == int:
                if self.pick_classes > len(set(self.class_list)):
                    print("ERROR: 'pick_classes' must be less than number of possible classes...")
                    return
                else:
                    random.shuffle(self.class_list)
                    self.class_list = self.class_list[:self.pick_classes]
            elif type(self.pick_classes) == list:
                if not all(elem in self.class_list for elem in self.pick_classes):
                    print("ERROR: Some class in 'pick_classes' not in dataset...")
                    return
                else:
                    self.class_list = self.pick_classes
            else:
                print("ERROR: Invalid 'pick_classes' argument...")
                return

        # load the selected classes from the dataset
        for idx, class_name in enumerate(self.class_list):
            class_dir = self.root + '/' + class_name
            for file in os.listdir(class_dir):
                image = cv2.imread(class_dir + '/' + file)
                self.data.append(image)
                self.labels.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]

        # returning data as PIL image is consistent with other PyTorch datasets
        data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, label

    # Description: this function will randomly sample 'per_class' examples from each class and concatenate them into a
    #   new dataset
    # Input:
    #   per_class: number of examples per class to sample from dataset
    #       -- if set to 'min' or an invalid number, value will be the minimum possible, as determined by the dataset
    def split_dataset(self, per_class='min'):
        num_classes = len(set(self.labels))
        class_counts = [self.labels.count(i) for i in range(num_classes)]

        if per_class == 'min':
            per_class = min(class_counts)
        elif per_class > min(class_counts):
            print("ERROR: not enough examples for class " + str(class_counts.index(min(class_counts))))
            print("Setting per_class to min(class_counts)...")
            per_class = min(class_counts)

        # shuffle X and Y in same order
        p = np.random.permutation(len(self.data))
        self.data = [self.data[i] for i in p]
        self.labels = [self.labels[i] for i in p]

        # define and create new dataset
        data_new = []
        labels_new = []

        for c in range(num_classes):
            iter = 0
            for p in range(per_class):
                while True:
                    if self.labels[iter] == c:
                        data_new.append(self.data[iter])
                        labels_new.append(self.labels[iter])
                        iter += 1
                        break
                    iter += 1

        # shuffle X_new and Y_new in same order
        p = np.random.permutation(len(data_new))
        data_new = [data_new[i] for i in p]
        labels_new = [labels_new[i] for i in p]

        # set new set as current dataset
        self.data = data_new
        self.labels = labels_new