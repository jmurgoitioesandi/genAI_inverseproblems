import numpy as np
import torch
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import generator_encdec_dense_2D, critic_dense_2D
from cwgan import Conditional_WGAN
from utils.data_utils import LoadImageDataset_JME
from utils.plotting_functions import (
    plotting_image_grid_cwgan_true_vs_generated,
)
from config import cla


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

    train_data = LoadImageDataset_JME(
        directory=PARAMS.dataset_directory, N=PARAMS.n_train, label="train"
    )

    val_data = LoadImageDataset_JME(
        directory=PARAMS.dataset_directory, N=PARAMS.n_val, label="val"
    )

    train_loader = DataLoader(
        train_data, batch_size=PARAMS.batch_size, shuffle=True, drop_last=True
    )

    val_loader = DataLoader(
        val_data, batch_size=PARAMS.batch_size, shuffle=False, drop_last=False
    )

    # Creating the models

    x_shape = (1, 64, 64)
    y_shape = (12, 64, 64)
    z_shape = (10, 1, 1)

    generator_model = generator_encdec_dense_2D(
        y_shape,
        out_channels=1,
        z_dim=PARAMS.z_dim,
        k0=16,
        act_param=0.1,
        denselayers=2,
        dense_int_out=12,
    )
    critic_model = critic_dense_2D(
        x_shape,
        y_shape,
        k0=16,
        denselayers=2,
        dense_int_out=16,
    )

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
        gp_coef=10,
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
