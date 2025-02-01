"""
Created on June 2021

@author: javiermurgoitio
"""

from fenics import *
from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import *
from mu_funcs_given import generate_mu_domain_withparams


def F_operator(mu_params):
    # ============== Parameters ======================
    omega_par = 2500  ### The frequency at which the equation is solved
    alpha_par = 0.00005  ### Parameter determining the friction (amplitude dissipation)

    """
    Calculates the dimensions of the region, so that it is im_size x im_size, and 1.75 x 1.75.
    Also it gives the index limits to be used to extract the region of interest from the broader
    region modelled (to avoid the effect of refletions).
    """
    im_size = 64  # 64x64
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

    """
    This function solves the variational form of the Helmholtz equation and returns the
    real and imaginary amplitudes in the region together with the shear modulus (all of
    them in numpy array format)
    """
    mu_init = generate_mu_domain_withparams(
        x_0, y_0, 1000, mu_params, H, W
    )  # scale unit/m, e.g. 1000 mm/m
    mu_dom = interpolate(mu_init, P)

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

    u_r = project(u.sub(0), P)
    u_i = project(u.sub(1), P)

    # REMEMBER THAT IF A SCALAR MAP IS GIVEN THE P FUNCTION SPACE SHOULD BE CHOSEN FOR COORDINATES,
    # AND THE V FUNCTION SPACE SHOULD BE CHOSE WHEN A VECTOR MAP IS GIVEN.
    u_r_array_list = u_r.vector().get_local()
    u_i_array_list = u_i.vector().get_local()
    xyz = P.tabulate_dof_coordinates()
    x = xyz[:, 0]
    y = xyz[:, 1]
    mu_array_list = mu_dom.vector().get_local()

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

    del v, u, u_i, u_r, v_i, v_r, xyz, x, y, F
    return mu_array[:, y1_idx:y2_idx, x1_idx:x2_idx, :], np.concatenate(
        (
            u_r_array[:, y1_idx:y2_idx, x1_idx:x2_idx, :],
            u_i_array[:, y1_idx:y2_idx, x1_idx:x2_idx, :],
        ),
        axis=3,
    )


def mu_from_params(mu_params):
    """
    Calculates the dimensions of the region, so that it is im_size x im_size, and 1.75 x 1.75.
    Also it gives the index limits to be used to extract the region of interest from the broader
    region modelled (to avoid the effect of refletions).
    """
    im_size = 64  # 64x64
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

    """
    This function solves the variational form of the Helmholtz equation and returns the
    real and imaginary amplitudes in the region together with the shear modulus (all of
    them in numpy array format)
    """

    mu_init = generate_mu_domain_withparams(
        x_0, y_0, 1000, mu_params, H, W
    )  # scale unit/m, e.g. 1000 mm/m
    mu_dom = interpolate(mu_init, P)

    xyz = P.tabulate_dof_coordinates()
    x = xyz[:, 0]
    y = xyz[:, 1]
    mu_array_list = mu_dom.vector().get_local()

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

    del x, y
    return mu_array[:, y1_idx:y2_idx, x1_idx:x2_idx, :]


def get_y(mu_vect):
    mu, y = F_operator(mu_vect.vector)
    mask = mu != 100
    mask = mask.astype(int)
    mask = np.repeat(mask, 2, axis=3)
    y = np.multiply(y, mask)
    return mu, y
