import numpy as np
import argparse
import os
from solver_funcs import F_operator, mu_from_params
import matplotlib.pyplot as plt
from utils import noise_addition_image

from posterior_funcs import Posterior, MuVector, Prior

parser = argparse.ArgumentParser(description="Sampling the posterior")
# parser.add_argument("--sigma_percentage", type=float, default=1.0)
# parser.add_argument("--directory", type=str, default="results")
parser.add_argument("--idx", type=str, default="results")
parser.add_argument("--sigma", type=float, default="results")
parser.add_argument("--epsilon", type=float, default="results")
args = parser.parse_args()

idx = args.idx

sigma_percentage = args.sigma
directory = f"results_{int(sigma_percentage*100)}percentnoise_1"

if not os.path.exists(directory):
    os.makedirs(directory)


def get_y(mu_vect):
    mu, y = F_operator(mu_vect.vector)
    mask = mu != 100
    mask = mask.astype(int)
    mask = np.repeat(mask, 2, axis=3)
    y = np.multiply(y, mask)
    return mu, y


mu_vect = MuVector()
_ = mu_vect.sample_vector()
mu_vect.vector = np.load("results_100percentnoise_1/mu_vect.npy")

mu, y = get_y(mu_vect)
sigma = y.max() * sigma_percentage
meas_y = noise_addition_image(y, noise_level=sigma)
np.save(directory + "/sigma.npy", np.array(y.max() * sigma_percentage))
np.save(directory + "/meas_y.npy", meas_y)
np.save(directory + "/mu_vect.npy", mu_vect.get_vector())

# fig, ax = plt.subplots(1, 3, figsize=(14, 4))

# cb = ax[0].imshow(mu[0, :, :, 0], vmax=250000)
# plt.colorbar(cb, ax=ax[0])
# cb = ax[1].imshow(meas_y[0, :, :, 0])
# plt.colorbar(cb, ax=ax[1])
# cb = ax[2].imshow(meas_y[0, :, :, 1])
# plt.colorbar(cb, ax=ax[2])

# plt.savefig(directory + "/mu_and_meas_y.png")
# plt.close()

## Sampling the posterior

sigma = np.load(directory + "/sigma.npy")
meas_y = np.load(directory + "/meas_y.npy")

chain = []
logpost = []
mu_chain = []

posterior = Posterior(sigma)
mu_vect = MuVector()
_ = mu_vect.sample_vector()
mu_cur = mu_vect.get_vector_4dim()


# mu_cur = np.load(directory + f"/chain_{idx}.npy")[-1, :]
# mu_vect.update_vector(mu_cur)
# chain_ = np.load(directory + f"/chain_{idx}.npy")
# logpost_ = np.load(directory + f"/logpost_{idx}.npy")
# mu_chain_ = np.load(directory + f"/mu_chain_{idx}.npy")
# for i in range(len(chain_)):
#     chain.append(chain_[i])
#     logpost.append(logpost_[i])
#     mu_chain.append(mu_chain_[i])

n_steps = 50000
# The following are the sampling parameters for noise as 100% of the maximum value of the image
epsilon = args.epsilon
factors = [4, 16, 40, 1]
np.save(directory + "/epsilon.npy", epsilon)
np.save(directory + "/factors.npy", np.array(factors))
# epsilon = np.load(directory + "/epsilon.npy")
# factors = np.load(directory + "/factors.npy")

n_accepted = 0

mu_dom_cur, y_cur = get_y(mu_vect)
p_cur = posterior.logpostprob(mu_cur, y_cur, meas_y)
chain.append(mu_cur)
logpost.append(p_cur)
mu_chain.append(mu_dom_cur)

for i in range(n_steps):
    mu_prop = mu_cur + mu_vect.get_step(epsilon, factors)
    mu_vect.update_vector(mu_prop)
    mu_dom_prop, y_prop = get_y(mu_vect)
    p_prop = posterior.logpostprob(mu_prop, y_prop, meas_y)
    alpha = min(1, np.exp(p_prop - p_cur))
    u = np.random.uniform()
    if i % 10 == 0:
        print(f"alpha: {alpha}; u: {u}, p_prop: {p_prop}, p_cur: {p_cur}")
    if u < alpha:
        mu_cur = mu_prop
        p_cur = p_prop
        mu_dom_cur = mu_dom_prop
        n_accepted += 1

    chain.append(mu_cur)
    logpost.append(p_cur)
    mu_chain.append(mu_dom_cur)

    if i % 10 == 0:
        print(f"Iteration: {i}; Acceptance rate: {n_accepted/(i+1)}")
        print(f"Current mu: {mu_vect.get_vector_4dim()}")

    if i % 100 == 0:
        np.save(directory + f"/chain_{idx}.npy", np.array(chain))
        np.save(directory + f"/mu_chain_{idx}.npy", np.array(mu_chain))
        np.save(directory + f"/logpost_{idx}.npy", np.array(logpost))


chain = np.array(chain)
np.save(directory + f"/chain_{idx}.npy", chain)
np.save(directory + f"/acceptance_rate_{idx}.npy", n_accepted / n_steps)
