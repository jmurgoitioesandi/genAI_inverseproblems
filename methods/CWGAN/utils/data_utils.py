import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import h5py
from skimage.color import rgb2yuv, yuv2rgb
import matplotlib.image as mpimg


class LoadImageDataset(Dataset):
    """
    Loading dataset which only contains Y channel data. Used for loading training data
    NOTE: For Loading RGB test/validation data, used LoadRGBDataset()
    """

    def __init__(self, datafile, divide=False, x_C=1, y_C=1, permute=False, N=1000):
        self.x_C = x_C
        self.y_C = y_C
        self.do_divide = divide
        self.data = self.load_data(datafile, N, permute)

    def __len__(self):
        if self.do_divide:
            return len(self.data[0])
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.data[idx]

        return x

    def load_data(self, datafile, N, permute=False):
        assert os.path.exists(datafile), (
            "Error: The data file " + datafile + " is unavailable."
        )

        if datafile.endswith(".npy"):
            data = np.load(datafile).astype(np.float32)

        elif datafile.endswith(".h5"):
            file = h5py.File(datafile, "r+")
            data = np.array(file["/images"][0::])
            file.close()

        totalN, H, W, channels = data.shape
        if N == None:
            n_use = totalN
        else:
            n_use = N
            assert totalN >= n_use

        data = torch.tensor(data[0:n_use, :, :, :].astype(np.float32))

        # Hacky permute
        if permute:
            data = data.permute(0, 3, 1, 2)

        print(f"     *** Datasets:")
        print(f"         ... samples loaded   = {n_use} of {totalN}")
        print(f"         ... sample dimension = {H}X{W}X{channels}")

        if self.do_divide:
            data_x = data[:, : self.x_C, :, :]
            data_y = data[:, self.x_C :, :, :]
            return data_x, data_y
        else:
            return data[:, : self.x_C, :, :]
