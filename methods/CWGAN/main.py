import numpy as np
import torch
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import get_models
from cwgan import Conditional_WGAN
from utils.plotting_functions import (
    plotting_image_grid_cwgan_true_vs_generated,
)
from config import cla
from datasets import get_dataset


def main():
    if torch.cuda.is_available():
        print("GPU available")
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    PARAMS = cla()

    np.random.seed(PARAMS.seed_no)
    torch.manual_seed(PARAMS.seed_no)

    train_dataset, val_dataset, test_dataset = get_dataset(PARAMS)

    train_loader = DataLoader(
        train_dataset, batch_size=PARAMS.batch_size, shuffle=True, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=PARAMS.batch_size, shuffle=False, drop_last=False
    )

    # Creating the models
    PARAMS.saving_dir = (
        PARAMS.saving_dir
        + "_"
        + PARAMS.problem
        + "_"
        + "CWGAN"
        + "_"
        + PARAMS.xtype
        + "_noiselvl"
        + str(int(100 * PARAMS.noiselvl))
    )

    generator_model, critic_model = get_models(PARAMS)

    # summary(
    #     generator_model,
    #     [
    #         y_shape,
    #         z_shape,
    #     ],
    # )
    # summary(critic_model, [x_shape, y_shape])

    g_optim = Adam(
        generator_model.parameters(),
        lr=PARAMS.learn_rate,
        betas=(0.5, 0.9),
    )
    c_optim = Adam(
        critic_model.parameters(),
        lr=PARAMS.learn_rate,
        betas=(0.5, 0.9),
    )

    wgan_trainer = Conditional_WGAN(
        batch_size=PARAMS.batch_size,
        directory=PARAMS.saving_dir,
        device=device,
        z_dim=PARAMS.z_dim,
        z_shape=[PARAMS.z_dim, 1, 1],
        gp_coef=PARAMS.gp_coef,
        n_critic=PARAMS.n_critic,
    )

    wgan_trainer.load_models_init(generator_model, critic_model, g_optim, c_optim)

    wgan_trainer.train(
        train_data=train_loader,
        val_data=val_loader,
        n_epoch=20000,
        sample_plotter=plotting_image_grid_cwgan_true_vs_generated,
    )


if __name__ == "__main__":
    main()
