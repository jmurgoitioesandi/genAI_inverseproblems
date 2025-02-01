import numpy as np
import argparse
import os
from solver_funcs import F_operator_HeatEquation
import matplotlib.pyplot as plt
from utils import noise_addition_image
from posterior_funcs import Posterior, MuVector

parser = argparse.ArgumentParser(description="Generate samples")
parser.add_argument("--idx", type=str, default="results")
parser.add_argument("--nsamples", type=int, default="number of samples")
args = parser.parse_args()

idx = args.idx
nsamples = args.nsamples

sigma_percentage = args.sigma
directory = f"/scratch1/murgoiti/HeatEquationSingleSource_GenerateSamples

if not os.path.exists(directory):
    os.makedirs(directory)

x_vect_list = []
y_list = []
x_list = []

for i in range(nsamples):
    x_vect = MuVector()
    x_vect.sample_vector()
    x_vect_list.append(x_vect)

    x, y = F_operator_HeatEquation(x_vect.get_vector())
    y_list.append(y)
    x_list.append(x)

np.save(f"{directory}/x_list_{idx}.npy", np.array(x_list))
np.save(f"{directory}/y_list_{idx}.npy", np.array(y_list))
np.save(f"{directory}/x_vect_list_{idx}.npy", np.array(x_vect_list))
