"""
Created on June 2021

@author: javiermurgoitio
"""

# from fenics import *
# from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import *
from generate_ONH_geom_randomuniform import *
from config import cla

PARAMS = cla()

# ============== Parameters ======================
omega_par = 2500  ### The frequency at which the equation is solved
alpha_par = 0.00005  ### Parameter determining the friction (amplitude dissipation)
N_train = 3000  ### Number of training samples to be produced.
N_valid = 0  ### Number of validation samples to be produced.
label = "20_21_22"

"""
Calculates the dimensions of the region, so that it is im_size x im_size, and 1.75 x 1.75.
Also it gives the index limits to be used to extract the region of interest from the broader
region modelled (to avoid the effect of refletions).
"""
im_size = 128  # 64x64
main_x_0 = 0
main_y_0 = 1.75
main_H = 1.75
main_W = 1.75
x_W = int(im_size * 2.5)
x_H = int(im_size * 3)
elem_size = 1.75 / (im_size - 1)
W = 1.75 + elem_size * im_size * 1.5
H = 1.75 + elem_size * im_size * 2  # in mm
x_0 = main_x_0 - (1.75 / (im_size - 1)) * im_size * 1.5
y_0 = main_y_0 - (1.75 / (im_size - 1)) * im_size
x1_idx = int(round((main_x_0 - x_0) / (W / (x_W - 1))))
x2_idx = int(x_W)
y2_idx = int(round(x_H - (main_y_0 - y_0) / (H / (x_H - 1))))
y1_idx = int(round(x_H - 1 - (main_H + main_y_0 - y_0) / (H / (x_H - 1))))

"""
The following part sets the mesh and the function spaces of the problem solved.
"""

scaling_factor = 0.001

# Create mesh and define function space
mesh = RectangleMesh(Point(0.0, 0.0), Point(W * 1e-3, H * 1e-3), x_W - 1, x_H - 1)

# Creating function spaces
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
P2 = FiniteElement("CG", mesh.ufl_cell(), 1)

element = MixedElement([P1, P2])
V = FunctionSpace(mesh, element)
P = FunctionSpace(mesh, "P", 1)

# Define boundary condition
tol = 1e-14


def right_boundary(x, on_boundary):
    """
    It defines the boundary.
    """
    return on_boundary and x[0] > W * scaling_factor - tol


def ONH_modelling():
    """
    This function solves the variational form of the Helmholtz equation and returns the
    real and imaginary amplitudes in the region together with the shear modulus (all of
    them in numpy array format)
    """

    mu_init, mask_init, mask_init_per_tissue = generate_mu_domain(
        x_0, y_0, 1000
    )  # scale unit/m, e.g. 1000 mm/m
    mu_dom = interpolate(mu_init, P)
    mask_dom = interpolate(mask_init, P)
    mask_per_tissue_dom = interpolate(mask_init_per_tissue, P)

    bc1 = DirichletBC(V.sub(0), Constant(0.00002), right_boundary)
    bc2 = DirichletBC(V.sub(1), Constant(0.00), right_boundary)
    bcs = [bc1, bc2]

    # Define variational problem
    v = TestFunction(V)
    v_r, v_i = split(v)
    u = Function(V)
    u_r, u_i = split(u)
    alpha = Constant(alpha_par)
    omega = 2 * pi * omega_par

    F = (
        -(omega**2) * u_r * v_r * dx
        + (mu_dom / 1000) * dot(grad(u_r), grad(v_r)) * dx
        - alpha * omega * (mu_dom / 1000) * dot(grad(u_i), grad(v_r)) * dx
        + -(omega**2) * u_i * v_i * dx
        + (mu_dom / 1000) * dot(grad(u_i), grad(v_i)) * dx
        + alpha * omega * (mu_dom / 1000) * dot(grad(u_r), grad(v_i)) * dx
    )

    solve(F == 0, u, bcs)

    u_r = project(u.sub(0) * mask_dom, P)
    u_i = project(u.sub(1) * mask_dom, P)

    # REMEMBER THAT IF A SCALAR MAP IS GIVEN THE P FUNCTION SPACE SHOULD BE CHOSEN FOR COORDINATES,
    # AND THE V FUNCTION SPACE SHOULD BE CHOSE WHEN A VECTOR MAP IS GIVEN.
    u_r_array_list = u_r.vector().get_local()
    u_i_array_list = u_i.vector().get_local()
    xyz = P.tabulate_dof_coordinates()
    x = xyz[:, 0]
    y = xyz[:, 1]
    mu_array_list = mu_dom.vector().get_local()
    mask_per_tissue_array_list = mask_per_tissue_dom.vector().get_local()

    u_r_array = fenics_2Darray_to_nparray(
        u_r_array_list,
        x,
        y,
        x_W,
        x_H,
        int(len(u_r_array_list) / (x_W * x_H)),
        W * scaling_factor,
        H * scaling_factor,
    )
    u_i_array = fenics_2Darray_to_nparray(
        u_i_array_list,
        x,
        y,
        x_W,
        x_H,
        int(len(u_i_array_list) / (x_W * x_H)),
        W * scaling_factor,
        H * scaling_factor,
    )

    mu_array = fenics_2Darray_to_nparray(
        mu_array_list,
        x,
        y,
        x_W,
        x_H,
        int(len(mu_array_list) / (x_W * x_H)),
        W * scaling_factor,
        H * scaling_factor,
    )
    mask_per_tissue_array = fenics_2Darray_to_nparray(
        mask_per_tissue_array_list,
        x,
        y,
        x_W,
        x_H,
        int(len(mask_per_tissue_array_list) / (x_W * x_H)),
        W * scaling_factor,
        H * scaling_factor,
    )

    del v, u, u_i, u_r, v_i, v_r, xyz, x, y, F
    return u_r_array, u_i_array, mu_array, mask_per_tissue_array


N_tot = N_train + 2 * N_valid
alpha_dir = int(1 / alpha_par)
savedir = (
    "exps/ONH_samples_freq_masks_" + str(omega_par) + "_invAlpha_" + str(alpha_dir)
)
noise_free = []

if not os.path.exists(savedir):
    os.makedirs(savedir)
else:
    print("\n     *** Folder already exists!\n")

"""
The following loop generates all the training, validation and testing samples.
"""

for idx in range(N_tot):
    print(idx)
    u_r, u_i, mu, tissue_mask = ONH_modelling()
    u_r = u_r[:, y1_idx:y2_idx, x1_idx:x2_idx, :]
    u_i = u_i[:, y1_idx:y2_idx, x1_idx:x2_idx, :]
    mu = mu[:, y1_idx:y2_idx, x1_idx:x2_idx, :]
    tissue_mask = tissue_mask[:, y1_idx:y2_idx, x1_idx:x2_idx, :]
    noise_free.append(np.concatenate((mu, u_r, u_i), 3))
    u_r = noise_addition_image(u_r, noise_level=1e-6)
    u_i = noise_addition_image(u_r, noise_level=1e-6)
    new_training_sample = np.concatenate((mu, u_r, u_i), 3)
    if idx == 0:
        traintest_images = new_training_sample
        tissue_masks = tissue_mask
    else:
        traintest_images = np.concatenate((traintest_images, new_training_sample), 0)
        tissue_masks = np.concatenate([tissue_masks, tissue_mask], axis=0)

# max_u_ri = np.amax(np.abs(traintest_images[:, :, :, 1:3]))
# max_mu = np.amax(traintest_images[:, :, :, 0])

### u_r and u_i are scaled to be between -1 and 1 and mu between 0 and 1. This improves the training of the network.

# traintest_images[:, :, :, 1:3] = traintest_images[:, :, :, 1:3] / max_u_ri
# traintest_images[:, :, :, 0] = traintest_images[:, :, :, 0] / max_mu

training = traintest_images[0:N_train]
validation = traintest_images[N_train : N_train + N_valid]
testing = traintest_images[N_train + N_valid : :]

store_images_hdf5(training, savedir, f"training_{label}")
store_images_hdf5(validation, savedir, f"validation_{label}")
store_images_hdf5(testing, savedir, f"testing_{label}")
store_images_hdf5(noise_free, savedir, f"noise_free_{label}")
store_images_hdf5(tissue_masks, savedir, f"tissue_masks_{label}")

"""
The values used to normalize the training samples are saved to use them to normalize the
actual medical data.
"""
# minmax_filename = savedir + "/norm_minmax_values.txt"

# with open(minmax_filename, "w") as f:
#     f.write("max u_ri value: " + str(max_u_ri))
#     f.write("\n")
#     f.write("max mu value: " + str(max_mu))

"""
u_t = []
dt = 0.000008
t=0d
N_t = 300
omega = 2*np.pi*2500.0

plt.imshow(u_r[0])
plt.show()

for idx in range(N_t):
    u_t.append(u_r[0]*np.cos(omega*t)-u_i[0]*np.sin(omega*t))
    u_t[idx][-1,-1,0] = -0.00002
    u_t[idx][0,-1,0] = 0.00002
    t = t+dt    
    print(t)

import imageio
imageio.mimwrite('ONH_1111kHz.mp4', u_t , fps = 40)
"""
