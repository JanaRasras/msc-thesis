""" A module for Dataset Preperation. """

import os
from typing import Optional # the value of this varilbe could be None or sth else

import torch # 
from torch import nn # the library that contains all NN
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from einops.layers.torch import Rearrange  # for reshaping

import numpy as np
from tqdm import tqdm

from phase4.utilities import TargetEncoderForMCML 
# from phase4.env import Environment


class StftDataset(Dataset):
    """ Create an STFT-DOA dataset pre-processed and ready for NN. """

    def __init__(self, folder: str,
                 tfx: nn.Module = None, tfy: nn.Module = None, # tfx/y: transform features/labels
                 n_steps: int = 1, many_to_one: bool = True):  # n_ inputs: seq_length defult value
        """ Load, prepare the dataset & ensure Batch is always the 1st dimension. """
        # Store parameters
        self.folder = folder
        self.tfx = tfx
        self.tfy = tfy
        self.n_steps = n_steps
        self.many_to_one = many_to_one

        # Load the data
        # weload the whole dataset into the GPU in order to speed up the get item function. This consumes some memory. (meaning: v. big models cant be loaded.)
        self.file_length = None # 50s/win 
        self.features, self.targets = self.setup() # 
        self.length = len(self.targets) - n_steps + 1 # the length of the whole ds

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]: # indx = example(each batch = 512 example,|| NN takes batch , 1 ex. (features and labels), features is a seq of frames, label is 1 frame. 1feature frame= array (1d vec) = fbins* 2* 2*mic ratio ) = indx of the label
                                                                            # 1 frame of label = one hot encod. of multi source doa
        """ Return a single examples of (x:float, y:long) for NN to process. 

        """
        features = self.features[index: index+self.n_steps] # seq_len

        labels = self.targets[index+self.n_steps-1] \
            if self.many_to_one \
            else self.targets[index: index+self.n_steps]   # indx: oldest(past) frame and not the current # -1 to select only the current label 

        return features.float(), labels.long()             #  float(32 bit to save memory) & int

    def __len__(self) -> int:
        """ Return the length of dataset while accounting for past frames. """
        return self.length

    def setup(self) -> tuple[torch.Tensor, torch.Tensor]:
        """ Load, process, and save the data as two arrays. """
        f_paths = [
            f"{self.folder}/{fname}"
            for fname in os.listdir(self.folder)
            if fname.endswith('.npz')
        ]
        assert len(f_paths) > 0, \
            f"Error: can't load the data because {self.folder} contains no .npz files!"

        all_features = []
        all_labels = []

        print(f"Reading the dataset from {self.folder}")
        for fname in tqdm(f_paths):
            data = np.load(fname)

            x = torch.from_numpy(data['x'])     # shape: (Freq ibns, nb. frames, channel input)
            y = torch.from_numpy(data['y'])     # shape: (nb. frames, nb of speakers)

            # Pre-processing
            all_features.append(    # NN dependant
                self.tfx(x) if self.tfx is not None else x
            )

            all_labels.append(      # One-hot-encoded
                self.tfy(y) if self.tfy is not None else y
            )

            if self.file_length is None:
                self.file_length = len(self.tfy(y))

        # Stack
        all_features = torch.vstack(all_features)
        all_labels = torch.vstack(all_labels).squeeze()

        return all_features, all_labels


class StftDoaDataModule(pl.LightningDataModule):
    """ .... """

    def __init__(self, folder: str,
                 batch_size: int, n_classes: int,
                 n_steps: int, many_to_one: bool,
                 download: bool):
        super().__init__()

        self.folder = folder
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.n_steps = n_steps
        self.download = download
        self.many_to_one = many_to_one

        self._n_labels = 4          # Max nb. of speakers
        self._resolution = 5
        self._silence_label = 999
        self._split_ratio = 0.8     # train/valid split
        self._cpus = 1

    def prepare_data(self) -> None:
        """ Run only once to generate/download the dataset. """
        if not self.download:
            return


    def setup(self, stage: Optional[str] = None):
        """ Load the data, apply transforms, and split into train/valid/test datasets. """
        # Create the Transforms
        feature_tf = Rearrange('F N R->N F R') #f: freq bins, R: # of ratios., N: N; nb of frames
                                               # We want the nb of examples to be at the begning..
        label_tf = TargetEncoderForMCML(
            self.n_classes, self._n_labels, self._resolution, self._silence_label)

        # Train stage
        if stage == 'fit':
            # Create the datasets
            train_valid_ds = StftDataset(f"{self.folder}/train",
                                         tfx=feature_tf, tfy=label_tf,
                                         n_steps=self.n_steps, many_to_one=self.many_to_one)

            # Split the dataset
            n_train = int(self._split_ratio * len(train_valid_ds))
            n_valid = len(train_valid_ds) - n_train

            train_ds, valid_ds = random_split(
                train_valid_ds, [n_train, n_valid])     # ir randomize the idxs of the dataset examples, not the dataset itself.(becaause geberating the seq. requires loading the data in a spesfix order {idx: idx+seq-len}

            # Store the dataset
            self.train_ds = train_ds
            self.valid_ds = valid_ds

        # Test stage
        if stage == 'test':
            test_ds = StftDataset(f"{self.folder}/test",
                                  tfx=feature_tf, tfy=label_tf,
                                  n_steps=self.n_steps, many_to_one=self.many_to_one)

            self.test_ds = test_ds

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size, shuffle=True, num_workers=self._cpus)

    def val_dataloader(self):
        return DataLoader(self.valid_ds,
                          batch_size=self.batch_size, shuffle=False, num_workers=self._cpus)

    def test_dataloader(self):
        # print(self.batch_size, self.test_ds.file_length, len(self.test_ds))
        # input('/') 
        return DataLoader(self.test_ds,
                          batch_size=len(self.test_ds), shuffle=False, num_workers=self._cpus)
