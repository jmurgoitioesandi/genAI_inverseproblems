# Import all the functions from DOLFIN (computational backend of the FEniCS project)
from fenics import *
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def interpolate_function(func, function_space):
    x = function_space.tabulate_dof_coordinates()[:, 0]
    y = function_space.tabulate_dof_coordinates()[:, 1]
    z = func.vector().get_local()
    xi = np.linspace(0.0, 1.0, 64)  # New x-coordinates
    yi = np.linspace(0.0, 1.0, 64)  # New y-coordinates
    xi, yi = np.meshgrid(xi, yi)  # Create a 2D grid
    zi = griddata(
        (x, y), z, (xi, yi), method="cubic"
    )  # Use 'cubic', 'linear', or 'nearest'
    return zi


def F_operator_HeatEquation(init_cond):
    # 20 is number of intervals Omega is divided into
    mesh = UnitSquareMesh(64, 64)
    # here interval is a FEniCS builtin representing a single interval
    elem = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, elem)

    kappa = Constant(0.25)  # physical material property
    S = Constant(0.0)  # source term

    dt = Constant(0.01)  # time step
    nb_t = 6  # number of time step - loop

    v = TestFunction(W)  # the test function
    # the TrialFunction is basically a symbol representing the unknown
    u = TrialFunction(W)
    u_old = Function(W)  # Solution at previous time step. Initialized to zero.

    a1 = init_cond[0]
    b1 = init_cond[1]
    c1 = init_cond[2]

    a2 = init_cond[3]
    b2 = init_cond[4]
    c2 = init_cond[5]

    P = FunctionSpace(mesh, "Lagrange", 1)

    class Kappa_dom(UserExpression):
        def __init__(self, a1, b1, c1, a2, b2, c2, **kwargs):
            super().__init__(**kwargs)
            self.x_c1 = a1
            self.y_c1 = b1
            self.radius1 = c1
            self.x_c2 = a2
            self.y_c2 = b2
            self.radius2 = c2

        def eval(self, value, x):
            if (x[0] - self.x_c1) ** 2 + (x[1] - self.y_c1) ** 2 < self.radius1**2:
                value[0] = 1.0
            elif (x[0] - self.x_c2) ** 2 + (x[1] - self.y_c2) ** 2 < self.radius2**2:
                value[0] = 1.0
            else:
                value[0] = 0.0

        def value_shape(self):
            return ()

    kappa_init = Kappa_dom(a1, b1, c1, a2, b2, c2, degree=0)
    u_old = interpolate(kappa_init, P)

    # plot(u_old)

    a = (u * v) / dt * dx + kappa * dot(
        grad(u), grad(v)
    ) * dx  # left hand side of our equation
    L = (u_old * v) / dt * dx + S * v * dx  # right hand side of our equation

    uh = Function(W)  # place to store the solution

    A = assemble(a)
    # for bc in bcs:
    #     bc.apply(A)

    solver = LUSolver(A)
    i = 0
    for t in range(nb_t):
        b = assemble(L)  # Reassemble L on every time step
        # for bc in bcs:  # RE-apply bc to b
        #     bc.apply(b)
        solver.solve(uh.vector(), b)  # Solve on given time step
        assign(u_old, uh)

        if i == 0:
            u_init = interpolate_function(uh, P)
        elif i == 5:
            u_end = interpolate_function(uh, P)

        i += 1

    return u_init, u_end
