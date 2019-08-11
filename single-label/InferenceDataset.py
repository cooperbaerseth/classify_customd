import os
from torch.utils.data import Dataset
from PIL import Image
import cv2

class InferenceDataset(Dataset):
    '''
    Loading inference data from the root folder
    '''

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data = []
        self.file_names = []
        self.grab_data()

    def grab_data(self):
        print("Processing Inference dataset...")

        # load inference images from root folder
        for img in os.listdir(self.root):
            self.file_names.append(img)
            self.data.append(cv2.imread(self.root + "/" + img))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        # returning data as PIL image is consistent with other PyTorch datasets
        data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        return data