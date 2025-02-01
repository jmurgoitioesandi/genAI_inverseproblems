import numpy as np
import math

# import random
from fenics import *
from dolfin import *

"""
THIS CODE GENERATES THE GEOMETRY OF THE OPTIC NERVE HEAD IN A 3.5x3.5 mm square
"""

# np.random.seed(seed=10)


class object_interior:
    def __init__(self, object_class):
        """
        The function_type is either circle, line or vertical.
        The parameters for the circle are: [x_center, y_center, radius]
        The parameters for the line are: [x_point, y_point, slope]
        """
        self.points_x = object_class.points_x
        self.points_y = object_class.points_y
        self.function1_type = object_class.function1_type
        self.function1_par = object_class.function1_par()
        self.function2_type = object_class.function2_type
        self.function2_par = object_class.function2_par()
        self.function3_type = object_class.function3_type
        self.function3_par = object_class.function3_par()
        self.function4_type = object_class.function4_type
        self.function4_par = object_class.function4_par()
        self.max_x = max(self.points_x)
        self.min_x = min(self.points_x)

    def function(self, x_pos, func_type, func_par):
        if func_type == "circle":
            return (
                func_par[3] * math.sqrt(func_par[2] ** 2 - (x_pos - func_par[0]) ** 2)
                + func_par[1]
            )
        elif func_type == "line":
            return (x_pos - func_par[0]) * func_par[2] + func_par[1]
        elif func_type == "line+cosine":
            """
            Parameters are the same as for the line plus: beginning of phase, end of phase, length of line in x coordinates, multiplicative factor
            """
            return (
                (x_pos - func_par[0]) * func_par[2]
                + func_par[1]
                + math.cos(
                    func_par[3]
                    + (func_par[4] - func_par[3]) * (x_pos - func_par[0]) / func_par[5]
                )
                * func_par[6]
                - math.cos(func_par[3]) * func_par[6]
            )

    def check_if_inside(self, x_coord, y_coord):
        y_coords_calc = []
        if x_coord < self.min_x:
            return False
        if x_coord > self.max_x:
            return False
        for idx in range(len(self.points_x)):
            if x_coord == self.points_x[idx]:
                y_coords_calc.append(self.points_y[idx])
        for idx in range(3):
            if (
                min(self.points_x[idx], self.points_x[idx + 1])
                < x_coord
                < max(self.points_x[idx], self.points_x[idx + 1])
            ):
                if idx == 0:
                    y_coords_calc.append(
                        self.function(x_coord, self.function1_type, self.function1_par)
                    )
                elif idx == 1:
                    y_coords_calc.append(
                        self.function(x_coord, self.function2_type, self.function2_par)
                    )
                elif idx == 2:
                    y_coords_calc.append(
                        self.function(x_coord, self.function3_type, self.function3_par)
                    )
        if (
            min(self.points_x[0], self.points_x[3])
            < x_coord
            < max(self.points_x[0], self.points_x[3])
        ):
            y_coords_calc.append(
                self.function(x_coord, self.function4_type, self.function4_par)
            )
        y_coords_calc.sort()
        if y_coords_calc[0] <= y_coord <= y_coords_calc[1]:
            return True
        else:
            return False


class lamina_cribrosa:
    def __init__(self):
        self.LC_x_1 = np.random.uniform(0.4, 1.2)
        self.LC_y_1 = np.random.uniform(2.38, 2.52)
        self.LC_x_2 = np.random.uniform(2.3, 3.1)
        self.LC_y_2 = self.LC_y_1 + np.random.uniform(-0.04, 0.04)
        self.LC_x_3 = self.LC_x_2 + np.random.uniform(-0.08, 0.08)
        self.LC_y_3 = np.random.uniform(2.68, 2.82)
        self.LC_x_4 = self.LC_x_1 + np.random.uniform(-0.08, 0.08)
        self.LC_y_4 = np.random.uniform(2.68, 2.82)
        self.radius_12 = np.random.uniform(3, 5)
        self.radius_34 = self.radius_12 + np.random.uniform(-0.1, 0.2)
        self.function1_type = "circle"
        self.function2_type = "line"
        self.function3_type = "circle"
        self.function4_type = "line"
        self.points_x = [self.LC_x_1, self.LC_x_2, self.LC_x_3, self.LC_x_4]
        self.points_y = [self.LC_y_1, self.LC_y_2, self.LC_y_3, self.LC_y_4]

    def function1_par(self):
        q1 = math.sqrt(
            (self.LC_x_2 - self.LC_x_1) ** 2 + (self.LC_y_2 - self.LC_y_1) ** 2
        )
        x3_1 = (self.LC_x_2 + self.LC_x_1) / 2
        z3_1 = (self.LC_y_2 + self.LC_y_1) / 2
        x_center = (
            x3_1
            - math.sqrt(self.radius_12**2 - (q1 / 2) ** 2)
            * (self.LC_y_1 - self.LC_y_2)
            / q1
        )
        y_center = (
            z3_1
            - math.sqrt(self.radius_12**2 - (q1 / 2) ** 2)
            * (self.LC_x_1 - self.LC_x_2)
            / q1
        )
        return [x_center, y_center, self.radius_12, -1]

    def function2_par(self):
        return [
            self.LC_x_2,
            self.LC_y_2,
            (self.LC_y_3 - self.LC_y_2) / (self.LC_x_3 - self.LC_x_2),
        ]

    def function3_par(self):
        q1 = math.sqrt(
            (self.LC_x_4 - self.LC_x_3) ** 2 + (self.LC_y_4 - self.LC_y_3) ** 2
        )
        x3_1 = (self.LC_x_4 + self.LC_x_3) / 2
        z3_1 = (self.LC_y_4 + self.LC_y_3) / 2
        x_center = (
            x3_1
            - math.sqrt(self.radius_34**2 - (q1 / 2) ** 2)
            * (self.LC_y_4 - self.LC_y_3)
            / q1
        )
        y_center = (
            z3_1
            - math.sqrt(self.radius_34**2 - (q1 / 2) ** 2)
            * (self.LC_x_4 - self.LC_x_3)
            / q1
        )
        return [x_center, y_center, self.radius_34, -1]

    def function4_par(self):
        return [
            self.LC_x_4,
            self.LC_y_4,
            (self.LC_y_4 - self.LC_y_1) / (self.LC_x_4 - self.LC_x_1),
        ]


class sclera_taper_left:
    def __init__(self, LC_y_1, LC_y_4, LC_x_1, LC_x_4):
        dist1 = np.random.uniform(0.01, 0.11)
        while dist1 < 0:
            dist1 = np.random.uniform(0.01, 0.11)
        dist2 = np.random.uniform(-0.11, -0.01)
        while dist2 > 0:
            dist2 = np.random.uniform(-0.11, -0.01)
        self.SCTL_y_3 = LC_y_4 + dist1
        self.SCTL_y_2 = LC_y_1 + dist2
        self.SCTL_x_3 = LC_x_4 + (self.SCTL_y_3 - LC_y_4) * (LC_x_4 - LC_x_1) / (
            LC_y_4 - LC_y_1
        )
        self.SCTL_x_2 = LC_x_1 + (self.SCTL_y_2 - LC_y_1) * (LC_x_4 - LC_x_1) / (
            LC_y_4 - LC_y_1
        )
        self.SCTL_x_4 = self.SCTL_x_3 - np.random.uniform(0.5, 0.7)
        self.SCTL_y_4 = self.SCTL_y_3 + np.random.uniform(0.0, 0.2)
        self.distance_41 = np.random.uniform(0.45, 1.15)
        while self.distance_41 < 0:
            self.distance_41 = np.random.uniform(0.45, 1.15)
        self.distance_34 = math.sqrt(
            (self.SCTL_x_4 - self.SCTL_x_3) ** 2 + (self.SCTL_x_4 - self.SCTL_x_3) ** 2
        )
        self.SCTL_x_1 = (
            self.SCTL_x_4
            + self.distance_41 * (self.SCTL_y_3 - self.SCTL_y_4) / self.distance_34
        )
        self.SCTL_y_1 = (
            self.SCTL_y_4
            - self.distance_41 * (self.SCTL_x_3 - self.SCTL_x_4) / self.distance_34
        )
        self.points_x = [self.SCTL_x_1, self.SCTL_x_2, self.SCTL_x_3, self.SCTL_x_4]
        self.points_y = [self.SCTL_y_1, self.SCTL_y_2, self.SCTL_y_3, self.SCTL_y_4]
        self.function1_type = "line"
        self.function2_type = "line"
        self.function3_type = "line"
        self.function4_type = "line"

    def function1_par(self):
        return [
            self.SCTL_x_1,
            self.SCTL_y_1,
            (self.SCTL_y_2 - self.SCTL_y_1) / (self.SCTL_x_2 - self.SCTL_x_1),
        ]

    def function2_par(self):
        return [
            self.SCTL_x_2,
            self.SCTL_y_2,
            (self.SCTL_y_3 - self.SCTL_y_2) / (self.SCTL_x_3 - self.SCTL_x_2),
        ]

    def function3_par(self):
        return [
            self.SCTL_x_3,
            self.SCTL_y_3,
            (self.SCTL_y_4 - self.SCTL_y_3) / (self.SCTL_x_4 - self.SCTL_x_3),
        ]

    def function4_par(self):
        return [
            self.SCTL_x_4,
            self.SCTL_y_4,
            (self.SCTL_y_4 - self.SCTL_y_1) / (self.SCTL_x_4 - self.SCTL_x_1),
        ]


class sclera_left:
    def __init__(self, SCT_x_1, SCT_y_1, SCT_x_4, SCT_y_4):
        self.SCL_x_2 = SCT_x_1
        self.SCL_y_2 = SCT_y_1
        self.SCL_x_3 = SCT_x_4
        self.SCL_y_3 = SCT_y_4
        self.radius_34 = np.random.uniform(8, 16)
        self.radius_12 = self.radius_34 + np.random.uniform(0.0, 2.0)
        self.distance_23 = math.sqrt(
            (self.SCL_x_2 - self.SCL_x_3) ** 2 + (self.SCL_y_2 - self.SCL_y_3) ** 2
        )
        self.center_y_34 = (
            self.SCL_y_3 + self.radius_34 * abs(SCT_y_4 - SCT_y_1) / self.distance_23
        )
        self.center_x_34 = (
            self.SCL_x_3 + self.radius_34 * (SCT_x_4 - SCT_x_1) / self.distance_23
        )
        self.center_y_12 = (
            self.SCL_y_2 + self.radius_12 * abs(SCT_y_4 - SCT_y_1) / self.distance_23
        )
        self.center_x_12 = (
            self.SCL_x_2 + self.radius_12 * (SCT_x_4 - SCT_x_1) / self.distance_23
        )
        self.SCL_x_1 = -3.5
        self.SCL_x_4 = -3.5
        while (
            abs(self.SCL_x_1 - self.center_x_12) > self.radius_12
            or abs(self.SCL_x_4 - self.center_x_34) > self.radius_34
        ):
            self.radius_34 = np.random.uniform(8, 16)
            self.radius_12 = self.radius_34 + np.random.uniform(0.0, 2.0)
            self.distance_23 = math.sqrt(
                (self.SCL_x_2 - self.SCL_x_3) ** 2 + (self.SCL_y_2 - self.SCL_y_3) ** 2
            )
            self.center_y_34 = (
                self.SCL_y_3
                + self.radius_34 * abs(SCT_y_4 - SCT_y_1) / self.distance_23
            )
            self.center_x_34 = (
                self.SCL_x_3 + self.radius_34 * (SCT_x_4 - SCT_x_1) / self.distance_23
            )
            self.center_y_12 = (
                self.SCL_y_2
                + self.radius_12 * abs(SCT_y_4 - SCT_y_1) / self.distance_23
            )
            self.center_x_12 = (
                self.SCL_x_2 + self.radius_12 * (SCT_x_4 - SCT_x_1) / self.distance_23
            )
        self.SCL_y_1 = self.center_y_12 - math.sqrt(
            self.radius_12**2 - (self.SCL_x_1 - self.center_x_12) ** 2
        )
        self.SCL_y_4 = self.center_y_34 - math.sqrt(
            self.radius_34**2 - (self.SCL_x_4 - self.center_x_34) ** 2
        )
        self.function1_type = "circle"
        self.function2_type = "line"
        self.function3_type = "circle"
        self.function4_type = "vertical"
        self.points_x = [self.SCL_x_1, self.SCL_x_2, self.SCL_x_3, self.SCL_x_4]
        self.points_y = [self.SCL_y_1, self.SCL_y_2, self.SCL_y_3, self.SCL_y_4]

    def function1_par(self):
        return [self.center_x_12, self.center_y_12, self.radius_12, -1]

    def function2_par(self):
        return [
            self.SCL_x_2,
            self.SCL_y_2,
            (self.SCL_y_3 - self.SCL_y_2) / (self.SCL_x_3 - self.SCL_x_2),
        ]

    def function3_par(self):
        return [self.center_x_34, self.center_y_34, self.radius_34, -1]

    def function4_par(self):
        return [self.SCL_x_4, self.SCL_y_4, 0]


class sclera_taper_right:
    def __init__(self, LC_y_2, LC_y_3, LC_x_2, LC_x_3):
        dist1 = np.random.uniform(0.1, 0.11)
        while dist1 < 0:
            dist1 = np.random.uniform(0.01, 0.11)
        dist2 = np.random.uniform(-0.11, -0.01)
        while dist2 > 0:
            dist2 = np.random.uniform(-0.11, -0.01)
        self.SCTR_y_4 = LC_y_3 + dist1
        self.SCTR_y_1 = LC_y_2 + dist2
        self.SCTR_x_4 = LC_x_3 + (self.SCTR_y_4 - LC_y_3) * (LC_x_3 - LC_x_2) / (
            LC_y_3 - LC_y_2
        )
        self.SCTR_x_1 = LC_x_2 + (self.SCTR_y_1 - LC_y_2) * (LC_x_3 - LC_x_2) / (
            LC_y_3 - LC_y_2
        )
        self.SCTR_x_3 = self.SCTR_x_4 + np.random.uniform(0.5, 0.7)
        self.SCTR_y_3 = self.SCTR_y_4 + np.random.uniform(0.0, 0.2)
        self.distance_32 = np.random.uniform(0.45, 1.15)
        self.distance_43 = math.sqrt(
            (self.SCTR_y_4 - self.SCTR_y_3) ** 2 + (self.SCTR_x_4 - self.SCTR_x_3) ** 2
        )
        self.SCTR_x_2 = (
            self.SCTR_x_3
            + self.distance_32 * (self.SCTR_y_3 - self.SCTR_y_4) / self.distance_43
        )
        self.SCTR_y_2 = (
            self.SCTR_y_3
            - self.distance_32 * (self.SCTR_x_3 - self.SCTR_x_4) / self.distance_43
        )
        self.points_x = [self.SCTR_x_1, self.SCTR_x_2, self.SCTR_x_3, self.SCTR_x_4]
        self.points_y = [self.SCTR_y_1, self.SCTR_y_2, self.SCTR_y_3, self.SCTR_y_4]
        self.function1_type = "line"
        self.function2_type = "line"
        self.function3_type = "line"
        self.function4_type = "line"

    def function1_par(self):
        return [
            self.SCTR_x_1,
            self.SCTR_y_1,
            (self.SCTR_y_2 - self.SCTR_y_1) / (self.SCTR_x_2 - self.SCTR_x_1),
        ]

    def function2_par(self):
        return [
            self.SCTR_x_2,
            self.SCTR_y_2,
            (self.SCTR_y_3 - self.SCTR_y_2) / (self.SCTR_x_3 - self.SCTR_x_2),
        ]

    def function3_par(self):
        return [
            self.SCTR_x_3,
            self.SCTR_y_3,
            (self.SCTR_y_4 - self.SCTR_y_3) / (self.SCTR_x_4 - self.SCTR_x_3),
        ]

    def function4_par(self):
        return [
            self.SCTR_x_4,
            self.SCTR_y_4,
            (self.SCTR_y_4 - self.SCTR_y_1) / (self.SCTR_x_4 - self.SCTR_x_1),
        ]


class sclera_right:
    def __init__(self, SCTR_x_2, SCTR_y_2, SCTR_x_3, SCTR_y_3):
        self.SCR_x_1 = SCTR_x_2
        self.SCR_y_1 = SCTR_y_2
        self.SCR_x_4 = SCTR_x_3
        self.SCR_y_4 = SCTR_y_3
        self.radius_34 = np.random.uniform(8, 16)
        self.radius_12 = self.radius_34 + np.random.uniform(0, 2)
        self.distance_41 = math.sqrt(
            (self.SCR_x_4 - self.SCR_x_1) ** 2 + (self.SCR_y_4 - self.SCR_y_1) ** 2
        )
        self.center_x_34 = (
            self.SCR_x_4
            + self.radius_34 * (self.SCR_x_4 - self.SCR_x_1) / self.distance_41
        )
        self.center_y_34 = (
            self.SCR_y_4
            + self.radius_34 * (self.SCR_y_4 - self.SCR_y_1) / self.distance_41
        )
        self.center_x_12 = (
            self.SCR_x_1
            + self.radius_12 * (self.SCR_x_4 - self.SCR_x_1) / self.distance_41
        )
        self.center_y_12 = (
            self.SCR_y_1
            + self.radius_12 * (self.SCR_y_4 - self.SCR_y_1) / self.distance_41
        )
        self.SCR_x_2 = 4.5
        self.SCR_x_3 = 4.5
        while (
            abs(self.SCR_x_1 - self.center_x_12) > self.radius_12
            or abs(self.SCR_x_4 - self.center_x_34) > self.radius_34
        ):
            self.radius_34 = np.random.uniform(8, 16)
            self.radius_12 = self.radius_34 + np.random.uniform(0, 2)
            self.distance_41 = math.sqrt(
                (self.SCR_x_4 - self.SCR_x_1) ** 2 + (self.SCR_y_4 - self.SCR_y_1) ** 2
            )
            self.center_x_34 = (
                self.SCR_x_4
                + self.radius_34 * (self.SCR_x_4 - self.SCR_x_1) / self.distance_41
            )
            self.center_y_34 = (
                self.SCR_y_4
                + self.radius_34 * (self.SCR_y_4 - self.SCR_y_1) / self.distance_41
            )
            self.center_x_12 = (
                self.SCR_x_1
                + self.radius_12 * (self.SCR_x_4 - self.SCR_x_1) / self.distance_41
            )
            self.center_y_12 = (
                self.SCR_y_1
                + self.radius_12 * (self.SCR_y_4 - self.SCR_y_1) / self.distance_41
            )
        self.SCR_y_2 = self.center_y_12 - math.sqrt(
            self.radius_12**2 - (self.SCR_x_2 - self.center_x_12) ** 2
        )
        self.SCR_y_3 = self.center_y_34 - math.sqrt(
            self.radius_34**2 - (self.SCR_x_3 - self.center_x_34) ** 2
        )
        self.function1_type = "circle"
        self.function2_type = "vertical"
        self.function3_type = "circle"
        self.function4_type = "line"
        self.points_x = [self.SCR_x_1, self.SCR_x_2, self.SCR_x_3, self.SCR_x_4]
        self.points_y = [self.SCR_y_1, self.SCR_y_2, self.SCR_y_3, self.SCR_y_4]

    def function1_par(self):
        return [self.center_x_12, self.center_y_12, self.radius_12, -1]

    def function2_par(self):
        return [self.SCR_x_2, self.SCR_y_2, 0]

    def function3_par(self):
        return [self.center_x_34, self.center_y_34, self.radius_34, -1]

    def function4_par(self):
        return [
            self.SCR_x_4,
            self.SCR_y_4,
            (self.SCR_y_4 - self.SCR_y_1) / (self.SCR_x_4 - self.SCR_x_1),
        ]


class upper_lamina_cribrosa:
    def __init__(
        self,
        LC_x_4,
        LC_y_4,
        LC_x_3,
        LC_y_3,
        SCTR_x_4,
        SCTR_y_4,
        SCTL_x_3,
        SCTL_y_3,
        function1_par,
    ):
        self.ULC_x_1 = LC_x_4
        self.ULC_y_1 = LC_y_4
        self.ULC_x_2 = LC_x_3
        self.ULC_y_2 = LC_y_3
        self.ULC_x_3 = SCTR_x_4
        self.ULC_y_3 = SCTR_y_4
        self.ULC_x_4 = SCTL_x_3
        self.ULC_y_4 = SCTL_y_3
        self.function1_type = "circle"
        self.function2_type = "line"
        self.function3_type = "line"
        self.function4_type = "line"
        self.function1_par1 = function1_par
        self.points_x = [self.ULC_x_1, self.ULC_x_2, self.ULC_x_3, self.ULC_x_4]
        self.points_y = [self.ULC_y_1, self.ULC_y_2, self.ULC_y_3, self.ULC_y_4]

    def function1_par(self):
        return self.function1_par1

    def function2_par(self):
        return [
            self.ULC_x_2,
            self.ULC_y_2,
            (self.ULC_y_3 - self.ULC_y_2) / (self.ULC_x_3 - self.ULC_x_2),
        ]

    def function3_par(self):
        return [
            self.ULC_x_3,
            self.ULC_y_3,
            (self.ULC_y_4 - self.ULC_y_3) / (self.ULC_x_4 - self.ULC_x_3),
        ]

    def function4_par(self):
        return [
            self.ULC_x_4,
            self.ULC_y_4,
            (self.ULC_y_4 - self.ULC_y_1) / (self.ULC_x_4 - self.ULC_x_1),
        ]


class lower_lamina_cribrosa:
    def __init__(
        self,
        SCTL_x_2,
        SCTL_y_2,
        SCTR_x_1,
        SCTR_y_1,
        LC_x_2,
        LC_y_2,
        LC_x_1,
        LC_y_1,
        function3_par,
    ):
        self.LLC_x_1 = SCTL_x_2
        self.LLC_y_1 = SCTL_y_2
        self.LLC_x_2 = SCTR_x_1
        self.LLC_y_2 = SCTR_y_1
        self.LLC_x_3 = LC_x_2
        self.LLC_y_3 = LC_y_2
        self.LLC_x_4 = LC_x_1
        self.LLC_y_4 = LC_y_1
        self.function1_type = "circle"
        self.function2_type = "line"
        self.function3_type = "circle"
        self.function4_type = "line"
        self.function3_par1 = function3_par
        self.radius_12 = function3_par[2]
        self.points_x = [self.LLC_x_1, self.LLC_x_2, self.LLC_x_3, self.LLC_x_4]
        self.points_y = [self.LLC_y_1, self.LLC_y_2, self.LLC_y_3, self.LLC_y_4]

    def function1_par(self):
        q1 = math.sqrt(
            (self.LLC_x_2 - self.LLC_x_1) ** 2 + (self.LLC_y_2 - self.LLC_y_1) ** 2
        )
        x3_1 = (self.LLC_x_2 + self.LLC_x_1) / 2
        z3_1 = (self.LLC_y_2 + self.LLC_y_1) / 2
        x_center = (
            x3_1
            - math.sqrt(self.radius_12**2 - (q1 / 2) ** 2)
            * (self.LLC_y_1 - self.LLC_y_2)
            / q1
        )
        y_center = (
            z3_1
            - math.sqrt(self.radius_12**2 - (q1 / 2) ** 2)
            * (self.LLC_x_1 - self.LLC_x_2)
            / q1
        )
        return [x_center, y_center, self.radius_12, -1]

    def function2_par(self):
        return [
            self.LLC_x_2,
            self.LLC_y_2,
            (self.LLC_y_3 - self.LLC_y_2) / (self.LLC_x_3 - self.LLC_x_2),
        ]

    def function3_par(self):
        return self.function3_par1

    def function4_par(self):
        return [
            self.LLC_x_4,
            self.LLC_y_4,
            (self.LLC_y_4 - self.LLC_y_1) / (self.LLC_x_4 - self.LLC_x_1),
        ]


class upper_ONH:
    def __init__(self, ULC_x_4, ULC_y_4, ULC_x_3, ULC_y_3, function1_par):
        self.UONH_x_1 = ULC_x_4
        self.UONH_y_1 = ULC_y_4
        self.UONH_x_2 = ULC_x_3
        self.UONH_y_2 = ULC_y_3
        self.function1_par1 = function1_par
        self.UONH_x_4 = ULC_x_4
        self.UONH_y_4 = ULC_y_4 + np.random.uniform(0.2, 0.4)
        self.UONH_x_3 = ULC_x_3
        self.UONH_y_3 = ULC_y_3 + np.random.uniform(0.2, 0.4)
        self.function1_type = "line"
        self.function2_type = "vertical"
        self.function3_type = "line+cosine"
        self.function4_type = "vertical"
        self.points_x = [self.UONH_x_1, self.UONH_x_2, self.UONH_x_3, self.UONH_x_4]
        self.points_y = [self.UONH_y_1, self.UONH_y_2, self.UONH_y_3, self.UONH_y_4]

    def function1_par(self):
        return self.function1_par1

    def function2_par(self):
        return [self.UONH_x_2, self.UONH_y_2, 0.0]

    def function3_par(self):
        return [
            self.UONH_x_4,
            self.UONH_y_4,
            (self.UONH_y_4 - self.UONH_y_3) / (self.UONH_x_4 - self.UONH_x_3),
            -np.pi / 3,
            np.pi * 2 + np.pi / 3,
            self.UONH_x_3 - self.UONH_x_4,
            np.random.uniform(0.04, 0.16),
        ]

    def function4_par(self):
        return [self.UONH_x_4, self.UONH_y_4, 0.0]


class retina_tapper_left:
    def __init__(
        self,
        UONH_x_1,
        UONH_y_1,
        UONH_x_4,
        UONH_y_4,
        SCTL_x_4,
        SCTL_y_4,
        funct1_par,
        funct2_par,
        funct4_par,
    ):
        self.RTL_x_2 = UONH_x_1
        self.RTL_y_2 = UONH_y_1
        self.RTL_x_3 = UONH_x_4
        self.RTL_y_3 = UONH_y_4
        self.RTL_x_1 = SCTL_x_4
        self.RTL_y_1 = SCTL_y_4
        self.function1_par1 = funct1_par
        self.function2_par1 = funct2_par
        self.RTL_y_4 = self.RTL_y_3
        self.RTL_x_4 = SCTL_x_4 + (self.RTL_y_4 - self.RTL_y_1) * (1 / funct4_par[2])
        self.function4_par2 = funct4_par[2]
        self.function1_type = "line"
        self.function2_type = "vertical"
        self.function3_type = "line"
        self.function4_type = "line"
        self.points_x = [self.RTL_x_1, self.RTL_x_2, self.RTL_x_3, self.RTL_x_4]
        self.points_y = [self.RTL_y_1, self.RTL_y_2, self.RTL_y_3, self.RTL_y_4]

    def function1_par(self):
        return self.function1_par1

    def function2_par(self):
        return self.function2_par1

    def function3_par(self):
        return [self.RTL_x_3, self.RTL_y_3, 0]

    def function4_par(self):
        return [self.RTL_x_4, self.RTL_y_4, self.function4_par2]


class retina_left:
    def __init__(
        self,
        RTL_x_1,
        RTL_y_1,
        RTL_x_4,
        RTL_y_4,
        SCL_x_4,
        SCL_y_4,
        function1_par,
        function2_par,
    ):
        self.RL_x_2 = RTL_x_1
        self.RL_y_2 = RTL_y_1
        self.RL_x_3 = RTL_x_4
        self.RL_y_3 = RTL_y_4
        self.RL_x_1 = SCL_x_4
        self.RL_y_1 = SCL_y_4
        self.function1_par1 = function1_par
        self.function2_par1 = function2_par
        self.center_x = function1_par[0]
        self.center_y = function1_par[1]
        self.radius_34 = math.sqrt(
            (self.center_x - self.RL_x_3) ** 2 + (self.center_y - self.RL_y_3) ** 2
        )
        self.RL_x_4 = self.RL_x_1
        while self.radius_34 < abs(self.RL_x_4 - self.center_x):
            self.radius_34 += 1.0
        self.RL_y_4 = self.center_y - math.sqrt(
            self.radius_34**2 - (self.RL_x_4 - self.center_x) ** 2
        )
        self.function1_type = "circle"
        self.function2_type = "line"
        self.function3_type = "circle"
        self.function4_type = "vertical"
        self.points_x = [self.RL_x_1, self.RL_x_2, self.RL_x_3, self.RL_x_4]
        self.points_y = [self.RL_y_1, self.RL_y_2, self.RL_y_3, self.RL_y_4]

    def function1_par(self):
        return self.function1_par1

    def function2_par(self):
        return self.function2_par1

    def function3_par(self):
        return [self.center_x, self.center_y, self.radius_34, -1]

    def function4_par(self):
        return [self.RL_x_4, self.RL_y_4, 0]


class retina_tapper_right:
    def __init__(
        self,
        UONH_x_2,
        UONH_y_2,
        UONH_x_3,
        UONH_y_3,
        SCTR_x_3,
        SCTR_y_3,
        function1_par,
        function4_par,
        function2_par,
    ):
        self.RTR_x_1 = UONH_x_2
        self.RTR_y_1 = UONH_y_2
        self.RTR_x_4 = UONH_x_3
        self.RTR_y_4 = UONH_y_3
        self.RTR_x_2 = SCTR_x_3
        self.RTR_y_2 = SCTR_y_3
        self.function1_par1 = function1_par
        self.function4_par1 = function4_par
        self.function2_par2 = function2_par[2]
        self.RTR_y_3 = self.RTR_y_4
        self.RTR_x_3 = self.RTR_x_2 + (self.RTR_y_3 - self.RTR_y_2) * (
            1 / function2_par[2]
        )
        self.function1_type = "line"
        self.function2_type = "line"
        self.function3_type = "line"
        self.function4_type = "vertical"
        self.points_x = [self.RTR_x_1, self.RTR_x_2, self.RTR_x_3, self.RTR_x_4]
        self.points_y = [self.RTR_y_1, self.RTR_y_2, self.RTR_y_3, self.RTR_y_4]

    def function1_par(self):
        return self.function1_par1

    def function2_par(self):
        return [self.RTR_x_2, self.RTR_y_2, self.function2_par2]

    def function3_par(self):
        return [self.RTR_x_3, self.RTR_y_3, 0]

    def function4_par(self):
        return self.function4_par1


class retina_right:
    def __init__(
        self,
        RTR_x_2,
        RTR_y_2,
        RTR_x_3,
        RTR_y_3,
        SCR_x_3,
        SCR_y_3,
        function1_par,
        function4_par,
    ):
        self.RR_x_1 = RTR_x_2
        self.RR_y_1 = RTR_y_2
        self.RR_x_4 = RTR_x_3
        self.RR_y_4 = RTR_y_3
        self.RR_x_2 = SCR_x_3
        self.RR_y_2 = SCR_y_3
        self.function1_par1 = function1_par
        self.function4_par1 = function4_par
        self.center_x = function1_par[0]
        self.center_y = function1_par[1]
        self.radius_34 = math.sqrt(
            (self.center_x - self.RR_x_4) ** 2 + (self.center_y - self.RR_y_4) ** 2
        )
        self.RR_x_3 = self.RR_x_2
        while self.radius_34 < abs(self.RR_x_3 - self.center_x):
            self.radius_34 += 1.0
        self.RR_y_3 = self.center_y - math.sqrt(
            self.radius_34**2 - (self.RR_x_3 - self.center_x) ** 2
        )
        self.function1_type = "circle"
        self.function2_type = "vertical"
        self.function3_type = "circle"
        self.function4_type = "line"
        self.points_x = [self.RR_x_1, self.RR_x_2, self.RR_x_3, self.RR_x_4]
        self.points_y = [self.RR_y_1, self.RR_y_2, self.RR_y_3, self.RR_y_4]

    def function1_par(self):
        return self.function1_par1

    def function2_par(self):
        return [self.RR_x_2, self.RR_y_2, 0]

    def function3_par(self):
        return [self.center_x, self.center_y, self.radius_34, -1]

    def function4_par(self):
        return self.function4_par1


class optic_nerve:
    def __init__(self, LLC_x_1, LLC_y_1, LLC_x_2, LLC_y_2, function3_par):
        self.ON_x_4 = LLC_x_1
        self.ON_y_4 = LLC_y_1
        self.ON_x_3 = LLC_x_2
        self.ON_y_3 = LLC_y_2
        self.function3_par1 = function3_par
        self.ON_y_1 = -1.5
        self.ON_y_2 = -1.5
        self.ON_x_1 = LLC_x_1 - np.random.uniform(0.55, 0.95)
        self.ON_x_2 = LLC_x_2 + np.random.uniform(0.55, 0.95)
        self.radius_23 = np.random.uniform(13, 17)
        self.radius_41 = np.random.uniform(13, 17)
        self.function1_type = "line"
        self.function2_type = "circle"
        self.function3_type = "circle"
        self.function4_type = "circle"
        self.points_x = [self.ON_x_1, self.ON_x_2, self.ON_x_3, self.ON_x_4]
        self.points_y = [self.ON_y_1, self.ON_y_2, self.ON_y_3, self.ON_y_4]

    def function1_par(self):
        return [self.ON_x_1, self.ON_y_1, 0.0]

    def function2_par(self):
        q1 = math.sqrt(
            (self.ON_x_3 - self.ON_x_2) ** 2 + (self.ON_y_3 - self.ON_y_2) ** 2
        )
        x3_1 = (self.ON_x_3 + self.ON_x_2) / 2
        z3_1 = (self.ON_y_3 + self.ON_y_2) / 2
        x_center = (
            x3_1
            - math.sqrt(self.radius_23**2 - (q1 / 2) ** 2)
            * (self.ON_y_3 - self.ON_y_2)
            / q1
        )
        y_center = (
            z3_1
            - math.sqrt(self.radius_23**2 - (q1 / 2) ** 2)
            * (self.ON_x_2 - self.ON_x_3)
            / q1
        )
        return [x_center, y_center, self.radius_23, +1]

    def function3_par(self):
        return self.function3_par1

    def function4_par(self):
        q1 = math.sqrt(
            (self.ON_x_4 - self.ON_x_1) ** 2 + (self.ON_y_4 - self.ON_y_1) ** 2
        )
        x3_1 = (self.ON_x_4 + self.ON_x_1) / 2
        z3_1 = (self.ON_y_4 + self.ON_y_1) / 2
        x_center = (
            x3_1
            - math.sqrt(self.radius_41**2 - (q1 / 2) ** 2)
            * (self.ON_y_1 - self.ON_y_4)
            / q1
        )
        y_center = (
            z3_1
            - math.sqrt(self.radius_41**2 - (q1 / 2) ** 2)
            * (self.ON_x_4 - self.ON_x_1)
            / q1
        )
        return [x_center, y_center, self.radius_41, +1]


class pia_matter_left:
    def __init__(self, ON_x_1, ON_y_1, ON_x_4, ON_y_4, function2_par):
        self.PML_x_2 = ON_x_1
        self.PML_y_2 = ON_y_1
        self.PML_x_3 = ON_x_4
        self.PML_y_3 = ON_y_4
        self.function2_par1 = function2_par
        self.width1 = np.random.uniform(0.06, 0.1)
        self.width2 = np.random.uniform(0.06, 0.1)
        self.PML_x_1 = ON_x_1 - self.width1
        self.PML_y_1 = ON_y_1
        self.PML_x_4 = ON_x_4 - self.width2
        self.PML_y_4 = ON_y_4 + 0.08
        self.radius_41 = function2_par[2] + (self.width2 + self.width1) * 0.5
        self.function1_type = "line"
        self.function2_type = "circle"
        self.function3_type = "line"
        self.function4_type = "circle"
        self.points_x = [self.PML_x_1, self.PML_x_2, self.PML_x_3, self.PML_x_4]
        self.points_y = [self.PML_y_1, self.PML_y_2, self.PML_y_3, self.PML_y_4]

    def function1_par(self):
        return [self.PML_x_1, self.PML_y_1, 0.0]

    def function2_par(self):
        return self.function2_par1

    def function3_par(self):
        return [
            self.PML_x_3,
            self.PML_y_3,
            (self.PML_y_3 - self.PML_y_4) / (self.PML_x_3 - self.PML_x_4),
        ]

    def function4_par(self):
        q1 = math.sqrt(
            (self.PML_x_4 - self.PML_x_1) ** 2 + (self.PML_y_4 - self.PML_y_1) ** 2
        )
        x3_1 = (self.PML_x_4 + self.PML_x_1) / 2
        z3_1 = (self.PML_y_4 + self.PML_y_1) / 2
        x_center = (
            x3_1
            - math.sqrt(self.radius_41**2 - (q1 / 2) ** 2)
            * (self.PML_y_1 - self.PML_y_4)
            / q1
        )
        y_center = (
            z3_1
            - math.sqrt(self.radius_41**2 - (q1 / 2) ** 2)
            * (self.PML_x_4 - self.PML_x_1)
            / q1
        )
        return [x_center, y_center, self.radius_41, +1]


class pia_matter_right:
    def __init__(self, ON_x_2, ON_y_2, ON_x_3, ON_y_3, function4_par):
        self.PMR_x_1 = ON_x_2
        self.PMR_y_1 = ON_y_2
        self.PMR_x_4 = ON_x_3
        self.PMR_y_4 = ON_y_3
        self.function4_par1 = function4_par
        self.width1 = np.random.uniform(0.06, 0.1)
        self.width2 = np.random.uniform(0.06, 0.1)
        self.PMR_x_2 = ON_x_2 + self.width1
        self.PMR_y_2 = ON_y_2
        self.PMR_x_3 = ON_x_3 + self.width2
        self.PMR_y_3 = ON_y_3 + 0.08
        self.radius_23 = function4_par[2] + (self.width2 + self.width1) * 0.5
        self.function1_type = "line"
        self.function2_type = "circle"
        self.function3_type = "line"
        self.function4_type = "circle"
        self.points_x = [self.PMR_x_1, self.PMR_x_2, self.PMR_x_3, self.PMR_x_4]
        self.points_y = [self.PMR_y_1, self.PMR_y_2, self.PMR_y_3, self.PMR_y_4]

    def function1_par(self):
        return [self.PMR_x_1, self.PMR_y_1, 0.0]

    def function2_par(self):
        q1 = math.sqrt(
            (self.PMR_x_3 - self.PMR_x_2) ** 2 + (self.PMR_y_3 - self.PMR_y_2) ** 2
        )
        x3_1 = (self.PMR_x_3 + self.PMR_x_2) / 2
        z3_1 = (self.PMR_y_3 + self.PMR_y_2) / 2
        x_center = (
            x3_1
            - math.sqrt(self.radius_23**2 - (q1 / 2) ** 2)
            * (self.PMR_y_3 - self.PMR_y_2)
            / q1
        )
        y_center = (
            z3_1
            - math.sqrt(self.radius_23**2 - (q1 / 2) ** 2)
            * (self.PMR_x_2 - self.PMR_x_3)
            / q1
        )
        return [x_center, y_center, self.radius_23, +1]

    def function3_par(self):
        return [
            self.PMR_x_3,
            self.PMR_y_3,
            (self.PMR_y_4 - self.PMR_y_3) / (self.PMR_x_4 - self.PMR_x_3),
        ]

    def function4_par(self):
        return self.function4_par1


def random_modulus_gen(mean, SD):
    output_mod = np.random.normal(mean, SD)
    while output_mod < 0 or output_mod > 2 * mean:
        output_mod = np.random.normal(mean, SD)
    return output_mod


def generate_mu_domain(x_0, y_0, scale):
    LC = lamina_cribrosa()
    inside_LC = object_interior(LC)

    SCTL = sclera_taper_left(
        LC.points_y[0], LC.points_y[3], LC.points_x[0], LC.points_x[3]
    )
    inside_SCTL = object_interior(SCTL)

    SCL = sclera_left(SCTL.SCTL_x_1, SCTL.SCTL_y_1, SCTL.SCTL_x_4, SCTL.SCTL_y_4)
    inside_SCL = object_interior(SCL)

    SCTR = sclera_taper_right(LC.LC_y_2, LC.LC_y_3, LC.LC_x_2, LC.LC_x_3)
    inside_SCTR = object_interior(SCTR)

    SCR = sclera_right(SCTR.SCTR_x_2, SCTR.SCTR_y_2, SCTR.SCTR_x_3, SCTR.SCTR_y_3)
    inside_SCR = object_interior(SCR)

    ULC = upper_lamina_cribrosa(
        LC.LC_x_4,
        LC.LC_y_4,
        LC.LC_x_3,
        LC.LC_y_3,
        SCTR.SCTR_x_4,
        SCTR.SCTR_y_4,
        SCTL.SCTL_x_3,
        SCTL.SCTL_y_3,
        LC.function3_par(),
    )
    inside_ULC = object_interior(ULC)

    LLC = lower_lamina_cribrosa(
        SCTL.SCTL_x_2,
        SCTL.SCTL_y_2,
        SCTR.SCTR_x_1,
        SCTR.SCTR_y_1,
        LC.LC_x_2,
        LC.LC_y_2,
        LC.LC_x_1,
        LC.LC_y_1,
        LC.function1_par(),
    )
    inside_LLC = object_interior(LLC)

    UONH = upper_ONH(
        ULC.ULC_x_4, ULC.ULC_y_4, ULC.ULC_x_3, ULC.ULC_y_3, ULC.function3_par()
    )
    inside_UONH = object_interior(UONH)

    RTL = retina_tapper_left(
        UONH.UONH_x_1,
        UONH.UONH_y_1,
        UONH.UONH_x_4,
        UONH.UONH_y_4,
        SCTL.SCTL_x_4,
        SCTL.SCTL_y_4,
        SCTL.function3_par(),
        UONH.function4_par(),
        SCTL.function4_par(),
    )
    inside_RTL = object_interior(RTL)

    RL = retina_left(
        RTL.RTL_x_1,
        RTL.RTL_y_1,
        RTL.RTL_x_4,
        RTL.RTL_y_4,
        SCL.SCL_x_4,
        SCL.SCL_y_4,
        SCL.function3_par(),
        RTL.function4_par(),
    )
    inside_RL = object_interior(RL)

    RTR = retina_tapper_right(
        UONH.UONH_x_2,
        UONH.UONH_y_2,
        UONH.UONH_x_3,
        UONH.UONH_y_3,
        SCTR.SCTR_x_3,
        SCTR.SCTR_y_3,
        SCTR.function3_par(),
        UONH.function2_par(),
        SCTR.function2_par(),
    )
    inside_RTR = object_interior(RTR)

    RR = retina_right(
        RTR.RTR_x_2,
        RTR.RTR_y_2,
        RTR.RTR_x_3,
        RTR.RTR_y_3,
        SCR.SCR_x_3,
        SCR.SCR_y_3,
        SCR.function3_par(),
        RTR.function2_par(),
    )
    inside_RR = object_interior(RR)

    ON = optic_nerve(
        LLC.LLC_x_1, LLC.LLC_y_1, LLC.LLC_x_2, LLC.LLC_y_2, LLC.function1_par()
    )
    inside_ON = object_interior(ON)

    PML = pia_matter_left(
        ON.ON_x_1, ON.ON_y_1, ON.ON_x_4, ON.ON_y_4, ON.function4_par()
    )
    inside_PML = object_interior(PML)

    PMR = pia_matter_right(
        ON.ON_x_2, ON.ON_y_2, ON.ON_x_3, ON.ON_y_3, ON.function2_par()
    )
    inside_PMR = object_interior(PMR)
    ####THIS MECHANICAL PROPERTIES HAVE BEEN TAKEN FROM SIGAL 2004 AND OTHER PAPERS BY SIGAL
    LC_SM = random_modulus_gen(73100, 46900)  # Zhang 2020
    SCL_SM = random_modulus_gen(
        125000, 50000
    )  # Qian2021, Youngs Modulus = 176.8 +- 14.3 at 6mmHg to 573.5 +- 64.4 at 30mmhg. I have use the mean of the two values as the average and a higher stdev just to give more variability.
    SCR_SM = random_modulus_gen(125000, 50000)
    UONH_SM = random_modulus_gen(9800, 3340)  # Sahan 2018
    RL_SM = random_modulus_gen(9800, 3340)
    RR_SM = random_modulus_gen(9800, 3340)
    ON_SM = random_modulus_gen(9800, 3340)
    PML_SM = random_modulus_gen(125000, 50000)
    PMR_SM = random_modulus_gen(125000, 50000)
    rotation_angle = np.random.uniform(-1, 1) * np.pi / 12
    rotation_array = np.array(
        [
            [math.cos(rotation_angle), -math.sin(rotation_angle)],
            [math.sin(rotation_angle), math.cos(rotation_angle)],
        ]
    )

    class mu_domains(UserExpression):
        def eval(self, value, x):
            point_to_check1 = (
                np.array([[x[0]], [x[1]]]) * scale
                + np.array([[x_0], [y_0]])
                - np.array([[W / 2], [H / 2]])
            )
            point_to_check2 = np.dot(rotation_array, point_to_check1)
            point_to_check3 = point_to_check2 + np.array([[W / 2], [H / 2]])
            if inside_LC.check_if_inside(point_to_check3[0, 0], point_to_check3[1, 0]):
                value[0] = LC_SM
            elif inside_SCTL.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = SCL_SM
            elif inside_SCL.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = SCL_SM
            elif inside_SCTR.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = SCR_SM
            elif inside_SCR.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = SCR_SM
            elif inside_ULC.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = UONH_SM
            elif inside_LLC.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = ON_SM
            elif inside_UONH.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = UONH_SM
            elif inside_RTL.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = RL_SM
            elif inside_RL.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = RL_SM
            elif inside_RTR.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = RR_SM
            elif inside_RR.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = RR_SM
            elif inside_ON.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = ON_SM
            elif inside_PML.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = PML_SM
            elif inside_PMR.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = PMR_SM
            else:
                value[0] = 100

    mu_init = mu_domains(degree=0)

    class mask_domains(UserExpression):
        def eval(self, value, x):
            point_to_check1 = (
                np.array([[x[0]], [x[1]]]) * scale
                + np.array([[x_0], [y_0]])
                - np.array([[W / 2], [H / 2]])
            )
            point_to_check2 = np.dot(rotation_array, point_to_check1)
            point_to_check3 = point_to_check2 + np.array([[W / 2], [H / 2]])
            if inside_LC.check_if_inside(point_to_check3[0, 0], point_to_check3[1, 0]):
                value[0] = 1.0
            elif inside_SCTL.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 1.0
            elif inside_SCL.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 1.0
            elif inside_SCTR.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 1.0
            elif inside_SCR.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 1.0
            elif inside_ULC.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 1.0
            elif inside_LLC.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 1.0
            elif inside_UONH.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 1.0
            elif inside_RTL.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 1.0
            elif inside_RL.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 1.0
            elif inside_RTR.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 1.0
            elif inside_RR.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 1.0
            elif inside_ON.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 1.0
            elif inside_PML.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 1.0
            elif inside_PMR.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 1.0
            else:
                value[0] = 0

    class mask_domains_per_tissue(UserExpression):
        def eval(self, value, x):
            point_to_check1 = (
                np.array([[x[0]], [x[1]]]) * scale
                + np.array([[x_0], [y_0]])
                - np.array([[W / 2], [H / 2]])
            )
            point_to_check2 = np.dot(rotation_array, point_to_check1)
            point_to_check3 = point_to_check2 + np.array([[W / 2], [H / 2]])
            if inside_LC.check_if_inside(point_to_check3[0, 0], point_to_check3[1, 0]):
                value[0] = 1.0
            elif inside_SCTL.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 2.0
            elif inside_SCL.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 2.0
            elif inside_SCTR.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 0.0
            elif inside_SCR.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 0.0
            elif inside_ULC.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 3.0
            elif inside_LLC.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 4.0
            elif inside_UONH.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 3.0
            elif inside_RTL.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 0.0
            elif inside_RL.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 0.0
            elif inside_RTR.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 0.0
            elif inside_RR.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 0.0
            elif inside_ON.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 4.0
            elif inside_PML.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 5.0
            elif inside_PMR.check_if_inside(
                point_to_check3[0, 0], point_to_check3[1, 0]
            ):
                value[0] = 0.0
            else:
                value[0] = 0

    mask_init = mask_domains(degree=0)
    mask_init_per_tissue = mask_domains_per_tissue(degree=0)

    return mu_init, mask_init, mask_init_per_tissue
