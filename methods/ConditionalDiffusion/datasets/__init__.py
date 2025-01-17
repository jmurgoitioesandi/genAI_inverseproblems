import os
import torch
import torchvision.transforms as transforms
import numpy as np
from datasets.phantoms import PhantomsDataset, PhantomsDatasetVal
from datasets.inverse_problems import (
    get_heat_eq_single_source_dataset,
    get_heat_eq_double_source_dataset,
    get_helmholtz_eq_dataset,
)


def get_dataset(args, config):
    if config.data.dataset == "HeatEqSingleSource":
        dataset, val_dataset, test_dataset = get_heat_eq_single_source_dataset(args)
    elif config.data.dataset == "HeatEqDoubleSource":
        dataset, val_dataset, test_dataset = get_heat_eq_double_source_dataset(args)
    elif config.data.dataset == "HelmholtzEq":
        dataset, val_dataset, test_dataset = get_helmholtz_eq_dataset(args)
    else:
        raise ValueError("Unknown dataset")
    return dataset, test_dataset, val_dataset


# def logit_transform(image, lam=1e-6):
#     image = lam + (1 - 2 * lam) * image
#     return torch.log(image) - torch.log1p(-image)

# def data_transform(config, X):
#     Add any data pre-processing here

# def inverse_data_transform(config, X):
#     Add any data post-processing here
