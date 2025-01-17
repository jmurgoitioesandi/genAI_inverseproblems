from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
import os


class PhantomsDataset(Dataset):
    """
    Loading dataset...
    """

    def __init__(self, directory, N=6000, use_factor=False):
        self.data_x, self.data_y = self.load_data(directory, N)
        self.use_factor = use_factor

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):

        if not self.use_factor:
            factor = 1
        else:
            mult = np.random.binomial(1, 0.5)
            factor = np.random.uniform(1.0, 2.0)
            if mult:
                factor = 1 / factor
        offset = np.random.randint(0, 3)
        return (
            self.data_x[idx],
            self.data_y[idx, offset : offset + 12, :, :]
            / torch.mean(torch.abs(self.data_y[idx, 0, :, :]))
            / factor,
        )

    def load_data(self, datafile, N):
        assert os.path.exists(datafile), (
            "Error: The data file " + datafile + " is unavailable."
        )

        data = np.load(datafile)
        data_x, data_y = data[:N, :1, :, :], data[:N, 1:, :, :]

        data_y = np.squeeze(data_y)

        data_x = torch.tensor(data_x[:N].astype(np.float32))
        data_y = torch.tensor(data_y[:N, :14, :, :].astype(np.float32))

        return data_x, data_y


class PhantomsDatasetVal(Dataset):
    """
    Loading dataset...
    """

    def __init__(self, directory, N=4):
        self.data_y = self.load_data(directory, N)

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        return self.data_y[idx]

    def load_data(self, datafile, N):
        assert os.path.exists(datafile), (
            "Error: The data file " + datafile + " is unavailable."
        )

        data = np.load(datafile)

        data_y = np.squeeze(data)
        data_y = torch.tensor(data_y[:N, :12, :, :].astype(np.float32))

        return data_y
