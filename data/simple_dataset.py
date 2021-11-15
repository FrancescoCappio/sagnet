import torch
import os
from PIL import Image
import numpy as np


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

def dataset_info_per_domain(txt_file, first_class=0):
    # like the one above, but divide per domain (first element of each line)
    with open(txt_file, 'r') as f:
        images_list = f.readlines()

    file_names = {}
    labels = {}

    for row in images_list:
        row = row.split(' ')

        name = row[0]
        lbl = int(row[1])

        domain = name.split('/')[0]
        if domain not in labels:
            labels[domain] = []
            file_names[domain] = []

        file_names[domain].append(name)
        labels[domain].append(lbl-first_class)

    return file_names, labels


def filter_k_shot(names, labels, k=5):
    np_names = np.array(names)
    np_labels = np.array(labels)
    np_indices = np.arange(len(labels))

    labels_set = set(np_labels.tolist())
    
    filtered_names = []
    filtered_labels = []
    for lbl in labels_set:
        mask = np_labels == lbl
        indices_lbl = np_indices[mask]
        if len(indices_lbl) > k:
            random_k_shots = np.random.choice(indices_lbl, k, replace=False)
        else:
            random_k_shots = indices_lbl
        filtered_names.extend(np_names[random_k_shots].tolist())
        filtered_labels.extend(np_labels[random_k_shots].tolist())
    return filtered_names, filtered_labels


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

