import torch
import numpy as np


def get_sigmas(config):
    if config.model.sigma_dist == "geometric":
        sigmas = (
            torch.tensor(
                np.exp(
                    np.linspace(
                        np.log(config.model.sigma_begin),
                        np.log(config.model.sigma_end),
                        config.model.num_classes,
                    )
                )
            )
            .float()
            .to(config.device)
        )
    elif config.model.sigma_dist == "uniform":
        sigmas = (
            torch.tensor(
                np.linspace(
                    config.model.sigma_begin,
                    config.model.sigma_end,
                    config.model.num_classes,
                )
            )
            .float()
            .to(config.device)
        )

    else:
        raise NotImplementedError("sigma distribution not supported")

    return sigmas


"""
    Langevin dynamics sampler
"""


@torch.no_grad()
def anneal_Langevin_dynamics(
    x_mod,
    y_mod,
    scorenet,
    sigmas,
    config,
    final_only=False,
    verbose=False,
):
    """Annealed Langevin dynamics sampler

    Inputs:
        x_mod (Torch tensor): Initial seed to the Markov chain
        y_mod (Torch tensor): Measurement vector/matrix; a realization of the measurement variable
        scorenet (Torch model): Score network
        sigmas (List): List of noise levels
        config (Dict): Configuration dictionary. Includes:
            n_steps_each: 'T' total number of steps to take at each noise level
            step_lr: '\epsilon' step size for the Langevin dynamics
            denoise (Bool): If True, denoise the final realization using the score network
        final_only (Bool): If True, only return the final realization
        verbose (Bool): If True, print the progress

    Returns:
        (Torch tensor) : List of realizations at each noise level or final realization
    """

    n_steps_each = config.sampling.n_steps_each
    step_lr = config.sampling.step_lr
    denoise = config.sampling.denoise

    images = []

    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                grad = scorenet(
                    x_mod,
                    y_mod.unsqueeze(0).repeat(x_mod.shape[0], 1, 1, 1),
                    labels,
                )

                noise = torch.randn_like(x_mod)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = np.sqrt(step_size / 2.0) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma**2

                if not final_only:
                    images.append(x_mod.to("cpu"))
                if verbose:
                    print(
                        "level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                            c,
                            step_size,
                            grad_norm.item(),
                            image_norm.item(),
                            snr.item(),
                            grad_mean_norm.item(),
                        )
                    )

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(
                x_mod.shape[0], device=x_mod.device
            )
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(
                x_mod, 
                y_mod.unsqueeze(0).repeat(x_mod.shape[0], 1, 1, 1),
                last_noise,
            )
            images.append(x_mod.to("cpu"))

        if final_only:
            return [x_mod.to("cpu")]
        else:
            return images
