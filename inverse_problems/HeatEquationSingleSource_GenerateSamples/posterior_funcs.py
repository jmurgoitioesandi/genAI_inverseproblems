import numpy as np
import matplotlib.pyplot as plt


class MuVector:

    def __init__(self):
        self.prior_params = {
            0: ((0.5, 0.3), "normal"),
            1: ((0.5, 0.3), "normal"),
            2: ((0.03, 0.15), "uniform"),
        }

        self.step_factors = np.zeros(len(self.prior_params))

        for i in range(len(self.prior_params)):
            if self.prior_params[i][1] == "uniform":
                self.step_factors[i] = (
                    self.prior_params[i][0][1] - self.prior_params[i][0][0]
                )
            elif self.prior_params[i][1] == "normal":
                self.step_factors[i] = self.prior_params[i][0][1] * 2

        self.vector = self.sample_mean_params()

    def sample_mean_params(self):
        params_list = []
        for i in range(len(self.prior_params)):
            if self.prior_params[i][1] == "uniform":
                params_list.append(
                    (self.prior_params[i][0][1] - self.prior_params[i][0][0]) / 2
                    + self.prior_params[i][0][0]
                )
            elif self.prior_params[i][1] == "normal":
                params_list.append(self.prior_params[i][0][0])
        return np.array(params_list)

    def update_vector(self, vector_update):
        self.vector = vector_update

    def sample_vector(self):
        vector_update = np.zeros(len(self.prior_params))
        for i in range(len(self.prior_params)):
            if self.prior_params[i][1] == "uniform":
                vector_update[i] = np.random.uniform(
                    self.prior_params[i][0][0], self.prior_params[i][0][1]
                )
            elif self.prior_params[i][1] == "normal":
                vector_update[i] = np.random.normal(
                    self.prior_params[i][0][0], self.prior_params[i][0][1]
                )
        self.update_vector(vector_update)
        return self.vector

    def get_vector(self):
        return self.vector

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
            0: ((0.0, 1.0), "uniform"),
            1: ((0.0, 1.0), "uniform"),
            2: ((0.03, 0.15), "uniform"),
        }

    def check_inside_prior(self, mu_vect):
        for i in range(len(mu_vect)):
            if self.prior_params[i][1] == "uniform":
                lims = self.prior_params[i][0]
            else:
                lims = (-np.inf, np.inf)
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
