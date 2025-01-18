import torch.nn as nn
from architecture_blocks import (
    UpSample2D,
    UpSample3D,
    DownSample2D,
    DownSample3D,
    DenseBlock2D,
    DenseBlock3D,
    ApplyNormalization2D,
    ApplyNormalization3D,
)
from torch import cat


def get_models(args):
    if args.model == "UNET_CIN"
        generator_model = generator_UNET_dense_2D(
            args.meas_channels,
            args.im_size,
            out_channels=1,
            z_dim=args.z_dim,
        )
        critic_model = critic_dense_2D(
            args.im_size,
            args.meas_channels,
            k0=16,
            denselayers=2,
            dense_int_out=16,
        )
    elif args.model == "ResNet_cat":
        raise NotImplementedError
    return generator_model, critic_model


class generator_UNET_dense_2D(nn.Module):
    """Generator model: U-Net with skip connections and Denseblocks
    input_x is assumed to have the shape (N, C, H, W)
    input_z is assumed to have the shape (N, z_dim, 1, 1) or None for Pix2Pix format
    """

    def __init__(
        self,
        meas_channels,
        im_size,
        out_channels=1,
        z_dim=50,
        k0=16,
        act_param=0.1,
        denselayers=2,
        dense_int_out=16,
        g_out="x",
    ):
        """
        x_shape does not include the number of samples N.
        """
        super(generator_UNET_dense_2D, self).__init__()
        C0, H0, W0 = meas_channels, im_size, im_size

        if z_dim == None:
            normalization = "in"
        else:
            normalization = "cin"

        # ------ Down branch -----------------------------
        H, W, k = H0, W0, k0
        self.d1 = DownSample2D(
            x_dim=C0, filters=k, downsample=False, act_param=act_param
        )
        self.d2 = DenseBlock2D(
            x_shape=(k, H, W),
            act_param=act_param,
            out_channels=dense_int_out,
            layers=denselayers,
        )

        self.d3 = DownSample2D(x_dim=k, filters=2 * k, act_param=act_param)
        H, W, k = (H - 2) // 2 + 1, (W - 2) // 2 + 1, 2 * k
        self.d4 = DenseBlock2D(
            x_shape=(k, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=dense_int_out * 2,
            layers=denselayers,
        )

        self.d5 = DownSample2D(x_dim=k, filters=2 * k, act_param=act_param)
        H, W, k = (H - 2) // 2 + 1, (W - 2) // 2 + 1, 2 * k
        self.d6 = DenseBlock2D(
            x_shape=(k, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=dense_int_out * 4,
            layers=denselayers,
        )

        self.d7 = DownSample2D(x_dim=k, filters=2 * k, act_param=act_param)
        H, W, k = (H - 2) // 2 + 1, (W - 2) // 2 + 1, 2 * k
        self.d8 = DenseBlock2D(
            x_shape=(k, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=dense_int_out * 8,
            layers=denselayers,
        )
        # ------------------------------------------------

        # ----- Base of UNet------------------------------
        self.base = DenseBlock2D(
            x_shape=(k, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=dense_int_out * 8,
            layers=denselayers,
        )
        # -------------------------------------------------

        # ------ Up branch -----------------------------
        self.u1 = UpSample2D(x_dim=k, filters=k // 2, act_param=act_param)
        H, W, k = 2 * H, 2 * W, k // 2
        self.u2 = DenseBlock2D(
            x_shape=(k, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=dense_int_out * 4,
            layers=denselayers,
        )

        self.u3 = UpSample2D(
            x_dim=k, filters=k // 2, concat=True, old_x_dim=k, act_param=act_param
        )
        H, W, k = 2 * H, 2 * W, k // 2
        self.u4 = DenseBlock2D(
            x_shape=(k, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=dense_int_out * 2,
            layers=denselayers,
        )

        self.u5 = UpSample2D(
            x_dim=k, filters=k // 2, concat=True, old_x_dim=k, act_param=act_param
        )

        H, W, k = 2 * H, 2 * W, k // 2
        self.u6 = DenseBlock2D(
            x_shape=(k, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=dense_int_out,
            layers=denselayers,
        )

        self.u7 = UpSample2D(
            x_dim=k,
            filters=k,
            concat=True,
            old_x_dim=k,
            upsample=False,
            act_param=act_param,
        )
        self.u8 = DenseBlock2D(
            x_shape=(k, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=dense_int_out,
            layers=denselayers,
        )

        self.u9 = UpSample2D(
            x_dim=k, filters=out_channels, upsample=False, activation=False
        )

        # ------------------------------------------------

    def forward(self, input_x, input_z=None):
        x1 = self.d1(input_x=input_x)
        x1 = self.d2(input_x=x1)

        x2 = self.d3(input_x=x1)
        x2 = self.d4(input_x=x2, input_z=input_z)

        x3 = self.d5(input_x=x2)
        x3 = self.d6(input_x=x3, input_z=input_z)

        x4 = self.d7(input_x=x3)
        x4 = self.d8(input_x=x4, input_z=input_z)

        x5 = self.base(input_x=x4, input_z=input_z)

        x6 = self.u1(input_x=x5)
        x6 = self.u2(input_x=x6, input_z=input_z)

        x7 = self.u3(input_x=x6, old_x=x3)
        x7 = self.u4(input_x=x7, input_z=input_z)

        x8 = self.u5(input_x=x7, old_x=x2)
        x8 = self.u6(input_x=x8, input_z=input_z)

        x9 = self.u7(input_x=x8, old_x=x1)
        x9 = self.u8(input_x=x9, input_z=input_z)

        x10 = self.u9(input_x=x9)

        output = x10

        return output


class critic_dense_2D(nn.Module):
    """Critic model using Denseblocks
    input_x and input_y are both assumed to have
    the shape (N, C, H, W)
    """

    def __init__(
        self, im_size=64, meas_channels=1, x_channels=1, k0=24, act_param=0.1, denselayers=2, dense_int_out=16
    ):
        """
        x_shape does not include the number of samples N.
        """
        super(critic_dense_2D, self).__init__()
        C0_x, H0, W0 = x_channels, im_size, im_size
        C0_y, H0, W0 = meas_channels, im_size, im_size

        # ------ Convolution layers -----------------------------
        H, W = H0, W0
        self.cnn1 = DownSample2D(
            x_dim=C0_x + C0_y, filters=k0, downsample=False, act_param=act_param
        )

        self.cnn2 = DenseBlock2D(
            x_shape=(k0, H, W),
            act_param=act_param,
            normalization="ln",
            out_channels=dense_int_out,
            layers=denselayers,
        )

        self.cnn3 = DownSample2D(
            x_dim=k0, filters=2 * k0, act_param=act_param, ds_k=4, ds_s=4
        )
        H, W = (H - 2) // 4 + 1, (W - 2) // 4 + 1
        self.cnn4 = DenseBlock2D(
            x_shape=(2 * k0, H, W),
            act_param=act_param,
            normalization="ln",
            out_channels=2 * dense_int_out,
            layers=denselayers,
        )

        self.cnn5 = DownSample2D(
            x_dim=2 * k0, filters=4 * k0, act_param=act_param, ds_k=4, ds_s=4
        )
        H, W = (H - 2) // 4 + 1, (W - 2) // 4 + 1

        self.cnn6 = DenseBlock2D(
            x_shape=(4 * k0, H, W),
            act_param=act_param,
            normalization="ln",
            out_channels=4 * k0,
            layers=denselayers,
        )

        self.cnn7 = DownSample2D(x_dim=4 * k0, filters=8 * k0, act_param=act_param)
        H, W = (H - 2) // 2 + 1, (W - 2) // 2 + 1
        self.cnn8 = DenseBlock2D(
            x_shape=(8 * k0, H, W),
            act_param=act_param,
            normalization="ln",
            out_channels=8 * dense_int_out,
            layers=denselayers,
        )

        # ----- Dense layers------------------------------
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(in_features=8 * k0 * H * W, out_features=128)
        self.LReLU = nn.ELU(alpha=act_param)
        self.LN = ApplyNormalization2D(x_shape=(128), normalization="ln")
        self.lin2 = nn.Linear(in_features=128, out_features=128)
        self.lin3 = nn.Linear(in_features=128, out_features=1)

        # ------------------------------------------------

    def forward(self, input_x, input_y):
        xy = cat((input_x, input_y), dim=1)

        x = self.cnn1(input_x=xy)
        x = self.cnn2(input_x=x)
        x = self.cnn3(input_x=x)
        x = self.cnn4(input_x=x)
        x = self.cnn5(input_x=x)
        x = self.cnn6(input_x=x)
        x = self.cnn7(input_x=x)
        x = self.cnn8(input_x=x)

        x = self.flat(x)
        x = self.lin1(x)
        x = self.LReLU(x)
        x = self.LN(x)
        x = self.lin2(x)
        x = self.LReLU(x)
        x = self.LN(x)
        output = self.lin3(x)

        return output