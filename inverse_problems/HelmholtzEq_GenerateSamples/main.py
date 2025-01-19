import numpy as np
import argparse
import os
from solver_funcs import F_operator, mu_from_params
import matplotlib.pyplot as plt
from utils import noise_addition_image

from posterior_funcs import Posterior, MuVector, Prior

parser = argparse.ArgumentParser(description="Generate samples")
parser.add_argument("--idx", type=str, default="results")
parser.add_argument("--nsamples", type=int, default="number of samples")
args = parser.parse_args()

idx = args.idx
nsamples = args.nsamples

sigma_percentage = args.sigma
directory = f"/scratch1/murgoiti/HelmholtzEq_GenerateSamples


def get_y(mu_vect):
    mu, y = F_operator(mu_vect.vector)
    mask = mu != 100
    mask = mask.astype(int)
    mask = np.repeat(mask, 2, axis=3)
    y = np.multiply(y, mask)
    return mu, y


mu_vect_list = []
mu_list = []
y_list = []

for i in range(nsamples):
    mu_vect = MuVector()
    _ = mu_vect.sample_vector()

    mu, y = get_y(mu_vect)

    mu_vect_list.append(mu_vect.vector)
    mu_list.append(mu)
    y_list.append(y)

np.save(f"{directory}/mu_list_{idx}.npy", np.array(mu_list))
np.save(f"{directory}/y_list_{idx}.npy", np.array(y_list))
np.save(f"{directory}/mu_vect_list_{idx}.npy", np.array(mu_vect_list))