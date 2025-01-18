import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import importlib
from torch.optim import Adam
from torch import (
    add,
    rand,
    randn,
    range,
    autograd,
    ones_like,
    pow,
    square,
    sqrt,
    sum,
    abs,
    mean,
    save,
    load,
    no_grad,
)
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class Conditional_WGAN:
    """
    WGAN class:
    Generator and critic need to be input to the class.
    """

    def __init__(
        self,
        batch_size=50,
        n_critic=4,
        gp_coef=10,
        z_dim=50,
        z_shape=[50, 1, 1],
        device=None,
        save_freq=500,
        directory="",
        learn_rate=1e-3,
        new=True,
    ):
        self.batch_size = batch_size
        self.n_critic = n_critic
        self.gp_coef = gp_coef
        self.z_dim = z_dim
        self.z_shape = z_shape
        self.generator = None
        self.critic = None
        self.g_optim = None
        self.c_optim = None
        self.device = device
        self.runs = 0
        self.learn_rate = learn_rate
        self.save_freq = save_freq
        self.epochs_trained = 0
        self.dir = directory
        self.last_g_dir = ""
        self.last_c_dir = ""
        self.last_g_optim_dir = ""
        self.last_c_optim_dir = ""
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        else:
            print("\n     *** Folder already exists!\n")

        if new:
            self.save_parameters()
        else:
            self.wgan_loader()

    def save_parameters(self):
        print("\n --- Saving parameters to file \n")
        param_file = self.dir + "/parameters.txt"
        param_dict = {
            "batch_size": self.batch_size,
            "n_critic": self.n_critic,
            "gp_coef": self.gp_coef,
            "z_dim": self.z_dim,
            "learn_rate": self.learn_rate,
            "save_freq": self.save_freq,
            "epochs_trained": self.epochs_trained,
            "runs": self.runs,
            "last_g_dir": self.last_g_dir,
            "last_c_dir": self.last_c_dir,
            "last_g_optim_dir": self.last_g_optim_dir,
            "last_c_optim_dir": self.last_c_optim_dir,
        }
        with open(param_file, "wb") as fid:
            pickle.dump(param_dict, fid)

    def load_models_init(self, generator, critic, g_optim, c_optim):
        self.generator = generator
        self.critic = critic
        self.generator.to(self.device)
        self.critic.to(self.device)
        self.g_optim = g_optim
        self.c_optim = c_optim

    def load_parameters(self):
        param_file = self.dir + "/parameters.txt"
        with open(param_file, "rb") as handle:
            data = handle.read()
        param_dict = pickle.loads(data)
        self.batch_size = param_dict["batch_size"]
        self.n_critic = param_dict["n_critic"]
        self.gp_coef = param_dict["gp_coef"]
        self.z_dim = param_dict["z_dim"]
        self.learn_rate = param_dict["learn_rate"]
        self.save_freq = param_dict["save_freq"]
        self.epochs_trained = param_dict["epochs_trained"]
        self.runs = param_dict["runs"]
        self.last_g_dir = param_dict["last_g_dir"]
        self.last_c_dir = param_dict["last_c_dir"]
        self.last_g_optim_dir = param_dict["last_g_optim_dir"]
        self.last_c_optim_dir = param_dict["last_c_optim_dir"]

    def wgan_loader(self):
        self.load_parameters()

        if self.runs > 0:
            self.generator = load(self.last_g_dir)
            self.critic = load(self.last_c_dir)

            self.g_optim = Adam(
                self.generator.parameters(),
                lr=self.learn_rate,
                betas=(0.9, 0.99),
            )
            self.c_optim = Adam(
                self.critic.parameters(),
                lr=self.learn_rate,
                betas=(0.9, 0.99),
            )

            g_optim_state = load(self.last_g_optim_dir)
            c_optim_state = load(self.last_c_optim_dir)
            self.g_optim.load_state_dict(g_optim_state)
            self.c_optim.load_state_dict(c_optim_state)

    def gradient_penalty(self, fake_X, true_X, true_Y, p=2, c0=1.0):
        """Evaluates full gradient penalty term"""
        batch_size, *other_dims = true_X.size()
        epsilon = rand([batch_size] + [1 for _ in range(0, len(other_dims))])
        epsilon = epsilon.expand(-1, *other_dims).to(self.device)
        x_hat = epsilon * true_X + (1 - epsilon) * fake_X
        x_hat.requires_grad = True
        true_Y.requires_grad = True
        c_hat = self.critic(x_hat, true_Y)
        grad = autograd.grad(
            outputs=c_hat,
            inputs=(x_hat, true_Y),
            grad_outputs=ones_like(c_hat).to(self.device),
            create_graph=True,
            retain_graph=True,
        )
        grad_x, grad_y = grad[0], grad[1]
        grad_x = grad_x.view(batch_size, -1)
        grad_y = grad_y.view(batch_size, -1)
        grad_norm = sqrt(
            1.0e-8 + add(sum(square(grad_x), dim=1), sum(square(grad_y), dim=1))
        )
        grad_penalty = pow(grad_norm - c0, p).mean()
        return grad_penalty

    def train_step_generator(self, true_X, true_Y):
        self.g_optim.zero_grad()
        z = randn([self.batch_size] + self.z_shape)
        z = z.to(self.device)
        fake_X = self.generator(true_Y, z)
        fake_val = self.critic(fake_X, true_Y)
        g_loss = -mean(fake_val)
        g_loss.backward()
        self.g_optim.step()
        missmatch = np.mean(
            np.sum(
                abs(true_X - fake_X).cpu().detach().numpy(),
                axis=(1, 2, 3),
            )
        )
        return g_loss.item(), missmatch.item()

    def eval_generator(self, true_X, true_Y):
        self.g_optim.zero_grad()
        z = randn([self.batch_size] + self.z_shape)
        z = z.to(self.device)
        fake_X = self.generator(true_Y, z)
        missmatch = np.mean(
            np.sum(
                abs(true_X - fake_X).cpu().detach().numpy(),
                axis=(1, 2, 3),
            )
        )
        return missmatch.item()

    def train_step_critic(self, true_X, true_Y):
        self.c_optim.zero_grad()
        z = randn([self.batch_size] + self.z_shape)
        z = z.to(self.device)
        fake_X = self.generator(true_Y, z).detach()
        fake_val = self.critic(fake_X, true_Y)
        true_val = self.critic(true_X, true_Y)
        gp_val = self.gradient_penalty(fake_X, true_X, true_Y)
        fake_loss = mean(fake_val)
        true_loss = mean(true_val)
        wd_loss = true_loss - fake_loss
        c_loss = -wd_loss + self.gp_coef * gp_val
        c_loss.backward()
        self.c_optim.step()
        return c_loss.item(), wd_loss.item()

    def cwgan_saver_while_training(self, log_dict, epoch):
        for key in log_dict.keys():
            np.savetxt(f"{self.dir}/{self.runs+1}/{key}.txt", log_dict[key])

        save(self.generator, f"{self.dir}/{self.runs+1}/generator_epoch={epoch}")
        save(self.critic, f"{self.dir}/{self.runs+1}/critic_epoch={epoch}")
        save(
            self.g_optim.state_dict(), f"{self.dir}/{self.runs+1}/g_optim_epoch={epoch}"
        )
        save(
            self.c_optim.state_dict(), f"{self.dir}/{self.runs+1}/c_optim_epoch={epoch}"
        )

        self.last_g_dir = f"{self.dir}/{self.runs+1}/generator_epoch={epoch}"
        self.last_c_dir = f"{self.dir}/{self.runs+1}/critic_epoch={epoch}"
        self.last_g_optim_dir = f"{self.dir}/{self.runs+1}/g_optim_epoch={epoch}"
        self.last_c_optim_dir = f"{self.dir}/{self.runs+1}/c_optim_epoch={epoch}"

    def train(
        self,
        train_data,
        val_data,
        n_epoch,
        verbose=True,
        verbose_freq=100,
        eval_freq=25,
        sample_plotter=None,
        n_samp_plot=4,
        n_stat=20,
    ):
        if os.path.exists(self.dir + f"/{self.runs+1}"):
            print("\n     *** Folder already exists!\n")
        else:
            os.makedirs(self.dir + f"/{self.runs+1}")
        c_loss_log = []
        g_loss_log = []
        wd_loss_log = []
        missmatch_train_log = []
        missmatch_val_log = []
        n_iters = 1
        for i in range(1, n_epoch + 1):
            for true_X, true_Y in train_data:
                true_X = true_X.to(self.device)
                true_Y = true_Y.to(self.device)

                # ----- Updating the critic -----
                c_loss, wd_loss = self.train_step_critic(true_X, true_Y)
                c_loss_log.append(c_loss)
                wd_loss_log.append(wd_loss)

                if n_iters % self.n_critic == 0:
                    g_loss, missmatch = self.train_step_generator(true_X, true_Y)
                    g_loss_log.append(g_loss)
                    missmatch_train_log.append(missmatch)

                n_iters += 1

                if verbose and n_iters % verbose_freq == 0:
                    print(
                        f"***** n iters: {n_iters};",
                        f" c_loss: {c_loss}; g_loss: {g_loss}",
                    )
            if i == 1 or i % eval_freq == 0:
                with no_grad():
                    for true_X, true_Y in val_data:
                        true_X = true_X.to(self.device)
                        true_Y = true_Y.to(self.device)
                        missmatch = self.eval_generator(true_X, true_Y)
                        missmatch_val_log.append(missmatch)

            if i % self.save_freq == 0 or i == 1:
                log_dict = dict()
                log_dict["g_loss"] = g_loss_log
                log_dict["c_loss"] = c_loss_log
                log_dict["wd_loss"] = wd_loss_log
                log_dict["missmatch_train"] = missmatch_train_log
                log_dict["missmatch_val"] = missmatch_val_log
                self.cwgan_saver_while_training(log_dict, i)
                self.save_parameters()
                if sample_plotter:
                    with no_grad():
                        z = randn([n_samp_plot * n_stat] + self.z_shape)
                        z = z.to(self.device)
                        fake = (
                            self.generator(
                                true_Y[:n_samp_plot].repeat_interleave(n_stat, dim=0), z
                            )
                            .cpu()
                            .detach()
                            .numpy()
                        )
                        sample_plotter(
                            fake,
                            true_Y.cpu().detach().numpy(),
                            true_X.cpu().detach().numpy(),
                            n_stat,
                            f"{self.dir}/{self.runs+1}/generated_samples_epoch={int(i)}",
                        )

        self.runs += 1
        self.epochs_trained += n_epoch
        self.save_parameters()
