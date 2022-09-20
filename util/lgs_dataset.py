import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from PIL import Image
import os
import pickle


class LGS1MDataset(Dataset):
    """LGS dataset."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        print("Initializing LGS dataset...")

        self.lgs_dataset = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        self.keys_list = self.lgs_dataset.end_clean.unique()
        print("There are", len(self.keys_list), "end leaves in total.")
        self.encode_dict, self.decode_dict = {}, {}
        for ind, key in enumerate(self.keys_list):
            self.encode_dict[key] = ind
            self.decode_dict[ind] = key

    def __len__(self):
        return len(self.lgs_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.lgs_dataset.iloc[idx, 0])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
            
        label = self.encode_dict[self.lgs_dataset.iloc[idx, -1]]
        return image, label


class LGS12MDataset(Dataset):
    """LGS dataset."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, pkl_file, root_dir, transform=None, dicts=None):
        """
        Args:
            pkl_file (string): Path to the pkl file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        print("Initializing LGS dataset...")

        with open(pkl_file, 'rb') as load_f:
            self.lgs_df = pickle.load(load_f)
        print("Done loading.")
        self.root_dir = root_dir
        self.transform = transform
        
        self.val_counts = self.lgs_df.clean_end_leaf.value_counts()
        print("There are", len(self.val_counts), "end leaves in total.")

        weights = 1 / self.val_counts
        weights = weights / weights.mean()
        self.sample_weights = np.array(weights[self.lgs_df['clean_end_leaf']])
        self.lgs_df['weight'] = self.sample_weights

        if dicts is None:
            self.encode_dict, self.decode_dict = {}, {}
            for ind, key in enumerate(self.val_counts.keys()):
                self.encode_dict[key] = ind
                self.decode_dict[ind] = key
        else:
            self.encode_dict = dicts['encode']
            self.decode_dict = dicts['decode']

    def __len__(self):
        return len(self.lgs_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name, clean_leaf, weight = tuple(self.lgs_df.iloc[idx, i] for i in (1, -2, -1))

        try:
            image = Image.open(os.path.join(self.root_dir, img_name))
            if self.transform:
                image = self.transform(image)
            label = self.encode_dict[clean_leaf]
            return image, label, weight
        except:
            print("An error occured when loading image", img_name)
            return None


class LGSFullDataset(Dataset):
    """LGS dataset."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, pkl_file, root_dir, transform=None):
        """
        Args:
            pkl_file (string): Path to the pkl file with the annotations.
            root_dir (string): Directory with all LGS images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        print("Initializing LGS dataset...")

        with open(pkl_file, 'rb') as load_f:
            self.lgs_df = pickle.load(load_f)
        print("Done loading.")
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.lgs_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.lgs_df.iloc[idx, 0]

        try:
            image = Image.open(os.path.join(self.root_dir, img_name))
            width, height = image.size
            if self.transform:
                image = self.transform(image)
            return image, width, height, idx
        except:
            print("An error occured when loading image", img_name)
            print("The index is", idx)
            return None


def lgs_collate(batch):
    batch = list(filter (lambda x: x is not None, batch)) # filter out all the Nones
    return torch.utils.data.dataloader.default_collate(batch)
