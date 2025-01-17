from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
import os


class ONH64(Dataset):
    """
    Create ONH64 dataset
    """

    def __init__(self, file_name, transform=None):
        """_summary_

        Args:
            file_name (String): Filename for the HDF5 file that contains the data arrays
            transform (torchvision.transform, optional): Any transformations that must be applied to the data. Defaults to None.
                                                         Same transform is applied to both the input (X's) and the output (Y's).
        """
        self.file_name = file_name
        self.file = h5py.File(self.file_name, "r")
        self.data = torch.tensor(
            (np.asarray(self.file["images"])[:, :, :, 0]).astype(np.float32)
        )
        self.data_y = torch.tensor(
            (np.asarray(self.file["images"])[:, :, :, 1:]).astype(np.float32)
        )
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data[idx, :, :].unsqueeze(0)
        image_y = 0.5 * ((self.data_y[idx, :, :, :]).permute(2, 0, 1) + 1.0)

        if self.transform:
            image = self.transform(image)
            image_y = self.transform(image_y)

        return image, image_y


class LoadImageDataset_EXT_JME(Dataset):
    """
    Loading dataset...
    """

    def __init__(self, directory, N=1000):
        self.data_x, self.data_y = self.load_data(directory, N)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):

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
