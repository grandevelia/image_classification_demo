import os
import torch
from torch.utils import data
from utils import load_image
import numpy as np


class ImagenetteLoader(data.Dataset):
    """
    A simple dataloader for imagenette
    folds: number of total folds for cross validation
    k: current fold
    """
    def __init__(
        self,
        root_folder,
        label_file=None,
        num_classes=10,
        split="train",
        transforms=None,
        folds=5,
        k=0
    ):
        assert split in ["train", "val", "test"]
        assert folds == -1 or k < folds
        self.root_folder = root_folder
        self.transforms = transforms
        self.n_classes = num_classes
        
        # concat all samples into list of tuples (filename, class_id)
        file_label_list = []
        if split in ["train", "val"] and folds > -1:
            # For cross validation, merge train and validation sets into one long list
            for s in ["train", "val"]:
                labels = os.listdir(f"{root_folder}/{s}")
                for label_id, label in enumerate(labels):
                    for fn in os.listdir(f"{root_folder}/{s}/{label}"):
                        file_label_list.append((f"{root_folder}/{s}/{label}/{fn}", label_id))
            
            # Train set gets (folds-1/folds) samples, validation set gets (1/folds) samples
            n_samples = len(file_label_list)
            n_start = int(n_samples * k / folds)
            n_end = int(n_samples * (k + 1) / folds)
            if split == "train":
                file_label_list = file_label_list[:n_start] + file_label_list[n_end:]
            elif split == "val":
                file_label_list = file_label_list[n_start:n_end]
        else:
            labels = os.listdir(f"{root_folder}/{split}")
            for label_id, label in enumerate(labels):
                for fn in os.listdir(f"{root_folder}/{split}/{label}"):
                    file_label_list.append((f"{root_folder}/{split}/{label}/{fn}", label_id))
        
        self.file_label_list = file_label_list

    def __len__(self):
        return len(self.file_label_list)

    def __getitem__(self, index):
        # load img and label
        filename, label_id = self.file_label_list[index]
        img = np.ascontiguousarray(load_image(filename))
        label = label_id

        # apply data augmentation
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

    def get_index_mapping(self):
        # load the train label file
        train_label_file = os.path.join(self.root_folder, "classnames.txt")
        if not os.path.exists(train_label_file):
            raise ValueError("Label file {:s} does not exist!".format(label_file))
        with open(train_label_file) as f:
            lines = f.readlines()

        # get the category names
        id_index_map = {}
        for label_id, line in enumerate(lines):
            label_name = line.rstrip("\n")
            id_index_map[label_id] = label_name

        return id_index_map
