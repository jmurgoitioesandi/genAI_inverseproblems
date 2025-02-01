import numpy as np
from solver_funcs import F_operator, mu_from_params
import matplotlib.pyplot as plt
from utils import noise_addition_image
from scipy.special import erf


class MuVector:
    def __init__(self):
        self.param_lims_dict = {
            0: (0.4, 1.2),
            1: (2.38, 2.52),
            2: (2.3, 3.1),
            3: (-0.04, 0.04),
            4: (-0.08, 0.08),
            5: (2.68, 2.82),
            6: (-0.08, 0.08),
            7: (2.68, 2.82),
            8: (1, 5),
            9: (-0.2, 0.2),
            10: (0.01, 0.11),
            11: (-0.11, -0.01),
            12: (0.5, 0.7),
            13: (0, 0.2),
            14: (0.45, 1.15),
            15: (8, 16),
            16: (0, 2),
            17: (0.2, 0.4),
            18: (0.2, 0.4),
            19: (0.04, 0.16),
            20: (0.55, 0.95),
            21: (0.55, 0.95),
            22: (13, 17),
            23: (13, 17),
            24: (0.06, 0.1),
            25: (0.06, 0.1),
            26: (0, 146200),
            27: (0, 250000),
            28: (0, 19600),
            29: (0, 19600),
            30: (0, 19600),
            31: (0, 250000),
            32: (-np.pi / 12, np.pi / 12),
        }

        self.prior_params = {
            0: ((0.4, 1.2), "uniform"),
            1: ((73100, 46900), "normal"),
            2: ((125000, 50000), "normal"),
            3: ((-np.pi / 12, np.pi / 12), "uniform"),
        }

        self.step_factors = np.zeros(len(self.prior_params))

        for i in range(len(self.prior_params)):
            if self.prior_params[i][1] == "uniform":
                self.step_factors[i] = (
                    self.prior_params[i][0][1] - self.prior_params[i][0][0]
                )
            elif self.prior_params[i][1] == "normal":
                self.step_factors[i] = self.prior_params[i][0][1]

        self.vector = self.sample_mean_params()

    def sample_mean_params(self):
        params_list = []
        for i in range(len(self.param_lims_dict)):
            params_list.append(
                (self.param_lims_dict[i][1] - self.param_lims_dict[i][0]) / 2
                + self.param_lims_dict[i][0]
            )
        return np.array(params_list)

    def update_vector(self, vector_update):
        self.vector[0] = vector_update[0]
        self.vector[26] = vector_update[1]
        self.vector[27] = vector_update[2]
        self.vector[32] = vector_update[3]

    def sample_vector(self):
        vector_update = np.zeros(4)
        for i in range(4):
            if self.prior_params[i][1] == "uniform":
                vector_update[i] = np.random.uniform(
                    self.prior_params[i][0][0], self.prior_params[i][0][1]
                )
            elif self.prior_params[i][1] == "normal":
                vector_update[i] = self.random_modulus_gen(
                    self.prior_params[i][0][0], self.prior_params[i][0][1]
                )
        self.update_vector(vector_update)
        return self.vector

    def get_vector(self):
        return self.vector

    def get_vector_4dim(self):
        return self.vector[[0, 26, 27, 32]]

    def random_modulus_gen(self, mean, SD):
        output_mod = np.random.normal(mean, SD)
        while output_mod < 0 or output_mod > 2 * mean:
            output_mod = np.random.normal(mean, SD)
        return output_mod

    def get_step(self, epsilon, factors):
        step = np.random.normal(0, epsilon, len(self.prior_params))
        return step * self.step_factors * np.array(factors)


class Prior:
    """
    Class to define the prior distribution of the parameters.
    This class should be used to calculate the prior probability of the parameters.
    It should check whether the parameters are inside the prior distribution and to
    return the sum of the logprior probabilities of those parameters that are normally
    distributed.
    """

    def __init__(self):

        self.prior_params = {
            0: ((0.4, 1.2), "uniform"),
            1: ((73100, 46900), "normal"),
            2: ((125000, 50000), "normal"),
            3: ((-np.pi / 12, np.pi / 12), "uniform"),
        }

    def calc_gn_cfd(self, x):
        return erf(x / np.sqrt(2)) / 2 + 0.5

    def calc_norm_ct(self, mean, std):
        return self.calc_gn_cfd(mean / std) - self.calc_gn_cfd(-mean / std)

    def check_inside_prior(self, mu_vect):
        for i in range(len(mu_vect)):
            if self.prior_params[i][1] == "uniform":
                lims = self.prior_params[i][0]
            else:
                lims = (0, 2 * self.prior_params[i][0][0])
            if mu_vect[i] < lims[0] or mu_vect[i] > lims[1]:
                return False
        return True

    def calc_logprob_prior(self, mu_vect):
        assert len(mu_vect) == len(
            self.prior_params
        ), f"Length of mu vector does not match. Expected {len(self.prior_params)} but got {len(mu_vect)}"
        logprob = 0
        for i in range(len(mu_vect)):
            if self.prior_params[i][1] == "uniform":
                continue
            logprob += (
                -0.5
                * (mu_vect[i] - self.prior_params[i][0][0]) ** 2
                / self.prior_params[i][0][1] ** 2
            ) / self.calc_norm_ct(
                self.prior_params[i][0][0], self.prior_params[i][0][1]
            )
        return logprob

    def logprob(self, mu_vect):
        assert len(mu_vect) == len(
            self.prior_params
        ), "Length of mu vector does not match"
        if not self.check_inside_prior(mu_vect):
            return -np.inf
        return self.calc_logprob_prior(mu_vect)


class Likelihood:

    def __init__(self, sigma):
        self.sigma = sigma

    def calc_loglikelihood(self, y_cur, meas_y):
        loglikelihood = -0.5 * np.sum(np.power(meas_y - y_cur, 2)) / self.sigma**2
        return loglikelihood


class Posterior:
    def __init__(self, sigma):
        self.prior = Prior()
        self.likelihood = Likelihood(sigma)

    def logpostprob(self, mu_vect, new_y, meas_y):
        return self.likelihood.calc_loglikelihood(new_y, meas_y) + self.prior.logprob(
            mu_vect
        )
