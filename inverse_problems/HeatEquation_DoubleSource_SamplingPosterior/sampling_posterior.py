import numpy as np
import argparse
import os
from solver_funcs import F_operator_HeatEquation
import matplotlib.pyplot as plt
from utils import noise_addition_image
from posterior_funcs import Posterior, MuVector

parser = argparse.ArgumentParser(description="Sampling the posterior")
# parser.add_argument("--sigma_percentage", type=float, default=1.0)
# parser.add_argument("--directory", type=str, default="results")
parser.add_argument("--idx", type=str, default="1")
parser.add_argument("--sigma", type=float, default=0.1)
parser.add_argument("--epsilon", type=float, default=0.01)
parser.add_argument("--nsamples", type=int, default=1000)
args = parser.parse_args()

idx = args.idx

directory = f"results_{int(args.sigma*100)}percentnoise_{idx}"

if not os.path.exists(directory):
    os.makedirs(directory)

x_vect = MuVector()
_ = x_vect.sample_vector()
x_vect.vector = np.load(
    f"results_{int(args.sigma*100)}percentnoise_{idx}" + "/x_vect.npy"
)
# print(x_vect.get_vector())

x, y = F_operator_HeatEquation(x_vect.get_vector())
meas_y = noise_addition_image(y, noise_level=args.sigma)
np.save(directory + "/meas_y.npy", meas_y)
np.save(directory + "/x_vect.npy", x_vect.get_vector())

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

cb = ax[0].imshow(x[:, :])
plt.colorbar(cb, ax=ax[0])
cb = ax[1].imshow(meas_y[:, :])
plt.colorbar(cb, ax=ax[1])

plt.savefig(directory + "/x_and_meas_y.png")
plt.close()

## Sampling the posterior

sigma = args.sigma
meas_y = np.load(directory + "/meas_y.npy")

x_vect_chain = []
logpost = []
x_chain = []

posterior = Posterior(sigma)
x_vect = MuVector()
_ = x_vect.sample_vector()
x_vect_cur = x_vect.get_vector()


# x_vect_cur = np.load(directory + f"/x_vect_chain_{idx}.npy")[-1, :]
# x_vect.update_vector(x_vect_cur)
# x_vect_chain_ = np.load(directory + f"/x_vect_chain_{idx}.npy")
# logpost_ = np.load(directory + f"/logpost_{idx}.npy")
# x_chain_ = np.load(directory + f"/x_chain_{idx}.npy")
# for i in range(len(x_chain_)):
#     x_vect_chain.append(x_vect_chain_[i])
#     logpost.append(logpost_[i])
#     x_chain.append(x_chain_[i])

n_steps = args.nsamples  # - len(x_chain_)
# The following are the sampling parameters for noise as 100% of the maximum value of the image
epsilon = args.epsilon
factors = [1, 1, 1, 1, 1, 1]
np.save(directory + "/epsilon.npy", epsilon)
np.save(directory + "/factors.npy", np.array(factors))
# epsilon = np.load(directory + "/epsilon.npy")
# factors = np.load(directory + "/factors.npy")

n_accepted = 0

x_cur, y_cur = F_operator_HeatEquation(x_vect_cur)
p_cur = posterior.logpostprob(x_vect_cur, y_cur, meas_y)
# x_vect_chain.append(x_vect_cur)
# logpost.append(p_cur)
# x_chain.append(x_cur)

for i in range(n_steps):
    x_vect_prop = x_vect_cur + x_vect.get_step(epsilon, factors)
    x_vect.update_vector(x_vect_prop)
    x_prop, y_prop = F_operator_HeatEquation(x_vect.get_vector())
    p_prop = posterior.logpostprob(x_vect_prop, y_prop, meas_y)
    alpha = min(1, np.exp(p_prop - p_cur))
    u = np.random.uniform()
    # if i % 10 == 0:
    #     print(f"alpha: {alpha}; u: {u}, p_prop: {p_prop}, p_cur: {p_cur}")
    if u < alpha:
        x_vect_cur = x_vect_prop
        p_cur = p_prop
        x_cur = x_prop
        n_accepted += 1

    x_vect_chain.append(x_vect_cur)
    logpost.append(p_cur)
    x_chain.append(x_cur)

    if i % 100 == 0:
        print(f"Iteration: {i}; Acceptance rate: {n_accepted/(i+1)}")

    if i % 10000 == 0:
        np.save(directory + f"/x_vect_chain_{idx}.npy", np.array(x_vect_chain))
        np.save(directory + f"/x_chain_{idx}.npy", np.array(x_chain))
        np.save(directory + f"/logpost_{idx}.npy", np.array(logpost))


x_chain = np.array(x_chain)
x_vect_chain = np.array(x_vect_chain)
np.save(directory + f"/x_chain_{idx}.npy", x_chain)
np.save(directory + f"/x_vect_chain_{idx}.npy", x_vect_chain)
np.save(directory + f"/acceptance_rate_{idx}.npy", n_accepted / n_steps)
