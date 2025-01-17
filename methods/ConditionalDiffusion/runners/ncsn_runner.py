import numpy as np
import tqdm
from losses.dsm import anneal_dsm_score_estimation
import torch.nn.functional as F
import logging
import torch
import os
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from models.ncsnv2 import NCSNv2
from models.cinmlp import CINMLP
from datasets import get_dataset
from losses import get_optimizer
from models import anneal_Langevin_dynamics, get_sigmas
from models.ema import EMAHelper

__all__ = ["NCSNRunner"]


def get_model(config):
    if config.data.x_type == "image":
        return NCSNv2(config).to(config.device)
    elif config.data.x_type == "vector":
        return CINMLP(config).to(config.device)
    else:
        raise NotImplementedError(
            "Model not implemented for x_type: {}".format(config.data.x_type)
        )
    # This module can be adapted to load multiple models depending on the dataset
    # For example:
    # if config.data.dataset == <dataset1> or config.data.dataset == <dataset2>:
    #     return NCSNv2(config).to(config.device)
    # elif config.data.dataset == <dataset3>:
    #     return NCSNv2Deepest(config).to(config.device)
    # elif config.data.dataset == <dataset4>:
    #     return NCSNv2Deeper(config).to(config.device)
    # Make sure you import all those models from models/ncsnv2.py


def get_sampler(config):
    if config.diff_method.type == "sbld":
        return anneal_Langevin_dynamics
    elif config.diff_method.type == "ddpm":
        return anneal_Langevin_dynamics
    else:
        raise NotImplementedError(
            "Sampler not implemented for diff_method: {}".format(
                config.diff_method.type
            )
        )


def get_loss_func(config):
    if config.diff_method.type == "sbld":
        return anneal_dsm_score_estimation
    elif config.diff_method.type == "ddpm":
        return anneal_dsm_score_estimation
    else:
        raise NotImplementedError(
            "Loss function not implemented for diff_method: {}".format(
                config.diff_method.type
            )
        )


class NCSNRunner:
    def __init__(self, args, config):
        """
        This class parses config files and command line arguments, and creates directories for logging and saving
        """
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, "samples")
        os.makedirs(args.log_sample_path, exist_ok=True)

    ## Training function
    def train(self):
        ## Load the datasets (training and testing)
        dataset, test_dataset, val_dataset = get_dataset(self.args, self.config)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            drop_last=True,
        )

        test_iter = iter(test_loader)
        self.config.input_dim = (
            self.config.data.image_size**2 * self.config.data.channels
        )

        ## Tensorboard logger
        tb_logger = self.config.tb_logger

        ## Load the model
        score = get_model(self.config)

        ## Get the loss function
        loss_func = get_loss_func(self.config)

        ## Get the sampler function
        sampler = get_sampler(self.config)

        ## Load the control image -- an instance of measurement vector/matrix that will be used to plot images as training progresses
        control_X, control_Y = val_dataset.__getitem__(self.config.data.control_id)
        control_Y = control_Y.to(self.config.device)

        # score = torch.nn.DataParallel(score) # Only necessary for multiGPU training

        ## Specify the optimizer
        optimizer = get_optimizer(self.config, score.parameters())

        start_epoch = 0
        step = 0

        ## Initialize exponential moving average (EMA) helper that will keep track of the moving average of the model parameters
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)

        ## Load the model from a checkpoint if specified. This option has not been tested for this project as of August 17, 2023
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "checkpoint.pth"))
            score.load_state_dict(states[0])
            ### Make sure we can resume with different eps
            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        ## Get the sigmas for the annealing process
        sigmas = get_sigmas(self.config)

        if self.config.training.log_all_sigmas:
            ### Commented out training time logging to save time.
            test_loss_per_sigma = [None for _ in range(len(sigmas))]

            def hook(loss, labels):
                # for i in range(len(sigmas)):
                #     if torch.any(labels == i):
                #         test_loss_per_sigma[i] = torch.mean(loss[labels == i])
                pass

            def tb_hook():
                # for i in range(len(sigmas)):
                #     if test_loss_per_sigma[i] is not None:
                #         tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                #                              global_step=step)
                pass

            def test_hook(loss, labels):
                for i in range(len(sigmas)):
                    if torch.any(labels == i):
                        test_loss_per_sigma[i] = torch.mean(loss[labels == i])

            def test_tb_hook():
                for i in range(len(sigmas)):
                    if test_loss_per_sigma[i] is not None:
                        tb_logger.add_scalar(
                            "test_loss_sigma_{}".format(i),
                            test_loss_per_sigma[i],
                            global_step=step,
                        )

        else:
            hook = test_hook = None

            def tb_hook():
                pass

            def test_tb_hook():
                pass

        ## Training loop
        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                score.train()
                step += 1  ## Each step is not to be confused with an epoch, rather each step is one mini-batch of training data

                # Obtain mini-batch of training data (input or X's) and correponding measurement vector/matrix (or Y's)
                X = X.to(self.config.device)
                y = y.to(self.config.device)

                # X = data_transform(self.config, X)

                ## Compute loss
                loss = loss_func(score, X, y, sigmas, self.config, None, hook)

                ## Tensorboard logging
                tb_logger.add_scalar("loss", loss, global_step=step)

                tb_hook()
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ## Update the EMA helper
                if self.config.model.ema:
                    ema_helper.update(score)

                if step >= self.config.training.n_iters:
                    return 0

                if step % 100 == 0:
                    ## Compute losses on test set after every 100 steps (or epochs) and log them
                    if self.config.model.ema:
                        test_score = ema_helper.ema_copy(score)
                    else:
                        test_score = score

                    test_score.eval()
                    try:
                        test_X, test_Y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_Y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_Y = test_Y.to(self.config.device)

                    with torch.no_grad():
                        test_dsm_loss = loss_func(
                            test_score,
                            test_X,
                            test_Y,
                            sigmas,
                            self.config,
                            None,
                            hook=test_hook,
                        )
                        tb_logger.add_scalar(
                            "test_loss", test_dsm_loss, global_step=step
                        )
                        test_tb_hook()
                        logging.info(
                            "step: {}, test_loss: {}".format(step, test_dsm_loss.item())
                        )

                        del test_score

                ## Checkpoints: save the model and samples every snapshot_freq steps
                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(
                            self.args.log_path, "checkpoint_{}.pth".format(step)
                        ),
                    )
                    torch.save(
                        states, os.path.join(self.args.log_path, "checkpoint.pth")
                    )

                    if self.config.training.snapshot_sampling:
                        if self.config.model.ema:
                            test_score = ema_helper.ema_copy(score)
                        else:
                            test_score = score

                        test_score.eval()

                        ## Random state will be affected because of sampling during training time.
                        if self.config.data.x_type == "image":
                            init_samples = torch.rand(
                                36,
                                self.config.data.channels,
                                self.config.data.image_size,
                                self.config.data.image_size,
                                device=self.config.device,
                            )
                        elif self.config.data.x_type == "vector":
                            init_samples = torch.rand(
                                36,
                                self.config.data.size,
                                device=self.config.device,
                            )
                        # init_samples = data_transform(self.config, init_samples)

                        all_samples = sampler(
                            init_samples,
                            control_Y,
                            test_score,
                            sigmas.cpu().numpy(),
                            self.config,
                            final_only=True,
                            verbose=True,
                        )

                        if self.config.data.x_type == "image":   
                            sample = all_samples[-1].view(
                                all_samples[-1].shape[0],
                                self.config.data.channels,
                                self.config.data.image_size,
                                self.config.data.image_size,
                            )
                        elif self.config.data.x_type == "vector":
                            sample = all_samples[-1].view(
                                all_samples[-1].shape[0],
                                self.config.data.size,
                            )

                        # sample = inverse_data_transform(self.config, sample)

                        torch.save(
                            sample,
                            os.path.join(
                                self.args.log_sample_path, "samples_{}.pth".format(step)
                            ),
                        )

                        if self.config.data.x_type == "image":
                            image_grid = make_grid(sample, 6)
                            save_image(
                                image_grid,
                                os.path.join(
                                    self.args.log_sample_path,
                                    "image_grid_{}.png".format(step),
                                ),
                            )

                            image_grid_tb = make_grid(
                                sample, 6, normalize=True, value_range=(0, 1)
                            )
                            image_grid_arr = (
                                image_grid_tb.mul(255)
                                .add_(0.5)
                                .clamp_(0, 255)
                                .to(torch.uint8)
                                .detach()
                                .cpu()
                            )
                            tb_logger.add_image(
                                "samples", image_grid_arr, global_step=step
                            )

                        del test_score
                        del all_samples

    ## Sampling function
    def sample(self):
        ## Load model from specified checkpoint or from the last checkpoint
        if self.config.sampling.ckpt_id is None:
            states = torch.load(
                os.path.join(self.args.log_path, "checkpoint.pth"),
                map_location=self.config.device,
            )
        else:
            states = torch.load(
                os.path.join(
                    self.args.log_path, f"checkpoint_{self.config.sampling.ckpt_id}.pth"
                ),
                map_location=self.config.device,
            )

        score = get_model(self.config)
        sampler = get_sampler(self.config)

        # score = torch.nn.DataParallel(score) # Only necessary for multiGPU training
        score.load_state_dict(states[0], strict=True)

        ## Load the exponential moving average (EMA) model if specified
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        score.eval()

        tx, txx, val_dataset = get_dataset(self.args, self.config)
        control_X, control_Y = val_dataset.__getitem__(self.config.sampling.control_id)
        control_Y = control_Y.to(self.config.device)

        ## Sampling loop -- each forward pass loop generates batch_size number of samples
        for j in tqdm.tqdm(
            range(self.config.sampling.sample_size // self.config.sampling.batch_size),
            total=self.config.sampling.sample_size // self.config.sampling.batch_size,
            desc="sampling posterior",
        ):

            ## Initial seed for the MCMC chain
            init_samples = torch.rand(
                self.config.sampling.batch_size,
                self.config.data.channels,
                self.config.data.image_size,
                self.config.data.image_size,
                device=self.config.device,
            )
            # init_samples = data_transform(self.config, init_samples)

            all_samples = sampler(
                init_samples,
                control_Y,
                score,
                sigmas,
                self.config,
                final_only=self.config.sampling.final_only,
                verbose=True,
            )

            if not self.config.sampling.final_only:
                ## If final_only is False, then all_samples of the MCMC trajectory are saved
                for i, sample in tqdm.tqdm(
                    enumerate(all_samples),
                    total=len(all_samples),
                    desc="saving image samples",
                ):
                    sample = sample.view(
                        sample.shape[0],
                        self.config.data.channels,
                        self.config.data.image_size,
                        self.config.data.image_size,
                    )

                    # sample = inverse_data_transform(self.config, sample)

                    # Naming convention for saving samples as image grids --- image_grid_<ckpt_id>_<batch_id>_<MCMC_step>.png
                    if self.config.data.x_type == "image":
                        image_grid = make_grid(
                            sample, int(np.sqrt(self.config.sampling.batch_size))
                        )
                        save_image(
                            image_grid,
                            os.path.join(
                                self.args.image_folder,
                                "image_grid_{}_{}_{}.png".format(
                                    self.config.sampling.ckpt_id, j, i
                                ),
                            ),
                        )
                    torch.save(
                        sample,
                        os.path.join(
                            self.args.image_folder,
                            "samples_{}_{}_{}.pth".format(
                                self.config.sampling.ckpt_id, j, i
                            ),
                        ),
                    )
            else:
                sample = all_samples[-1].view(
                    all_samples[-1].shape[0],
                    self.config.data.channels,
                    self.config.data.image_size,
                    self.config.data.image_size,
                )

                # sample = inverse_data_transform(self.config, sample)

                ## Naming convention for saving samples as image grids --- image_grid_<ckpt_id>_<batch_id>.png
                if self.config.data.x_type == "image":
                    image_grid = make_grid(
                        sample, int(np.sqrt(self.config.sampling.batch_size))
                    )
                    save_image(
                        image_grid,
                        os.path.join(
                            self.args.image_folder,
                            "image_grid_{}_{}.png".format(
                                self.config.sampling.ckpt_id, j
                            ),
                        ),
                    )
                torch.save(
                    sample,
                    os.path.join(
                        self.args.image_folder,
                        "samples_{}_{}.pth".format(self.config.sampling.ckpt_id, j),
                    ),
                )
