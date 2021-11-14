import torch
import os
from PIL import Image


def dataset_info(txt_file, first_class=0):
    with open(txt_file, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []

    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1])-first_class)

    return file_names, labels


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, path_dataset, names, labels, transforms=None):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):

        framename = os.path.join(self.data_path, self.names[index])
        img = Image.open(framename).convert('RGB')

        if self.transforms is not None: 
            img = self.transforms(img)

        return img, self.labels[index]

    def __len__(self):
        return len(self.names)

