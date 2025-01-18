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


class generator_UNET_dense_2D(nn.Module):
    """Generator model: U-Net with skip connections and Denseblocks
    input_x is assumed to have the shape (N, C, H, W)
    input_z is assumed to have the shape (N, z_dim, 1, 1) or None for Pix2Pix format
    """

    def __init__(
        self,
        y_shape,
        out_channels=1,
        z_dim=50,
        k0=20,
        act_param=0.1,
        denselayers=3,
        dense_int_out=16,
        g_out="x",
    ):
        """
        x_shape does not include the number of samples N.
        """
        super(generator_UNET_dense_2D, self).__init__()
        C0, H0, W0 = y_shape

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


class generator_encdec_dense_2D(nn.Module):
    """Generator model: U-Net with skip connections and Denseblocks
    input_x is assumed to have the shape (N, C, H, W)
    input_z is assumed to have the shape (N, z_dim, 1, 1) or None for Pix2Pix format
    """

    def __init__(
        self,
        y_shape,
        out_channels=1,
        z_dim=50,
        k0=20,
        act_param=0.1,
        denselayers=3,
        dense_int_out=16,
        g_out="x",
    ):
        """
        x_shape does not include the number of samples N.
        """
        super(generator_encdec_dense_2D, self).__init__()
        C0, H0, W0 = y_shape

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

        self.u3 = UpSample2D(x_dim=k, filters=k // 2, act_param=act_param)
        H, W, k = 2 * H, 2 * W, k // 2
        self.u4 = DenseBlock2D(
            x_shape=(k, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=dense_int_out * 2,
            layers=denselayers,
        )

        self.u5 = UpSample2D(x_dim=k, filters=k // 2, act_param=act_param)

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

        x7 = self.u3(input_x=x6)
        x7 = self.u4(input_x=x7, input_z=input_z)

        x8 = self.u5(input_x=x7)
        x8 = self.u6(input_x=x8, input_z=input_z)

        x9 = self.u7(input_x=x8)
        x9 = self.u8(input_x=x9, input_z=input_z)

        x10 = self.u9(input_x=x9)

        output = x10

        return output


class generator_UNET_dense_3D_to_2D(nn.Module):
    """Generator model: U-Net with skip connections and Denseblocks
    input_x is assumed to have the shape (N, C, D, H, W)
    input_z is assumed to have the shape (N, z_dim, 1, 1, 1) or None for Pix2Pix format
    """

    def __init__(
        self,
        y_shape,
        out_channels=1,
        z_dim=50,
        k0=20,
        act_param=0.1,
        denselayers=3,
        dense_int_out=16,
        g_out="x",
    ):
        """
        x_shape does not include the number of samples N.
        """
        super(generator_UNET_dense_3D_to_2D, self).__init__()
        C0, D0, H0, W0 = y_shape

        if z_dim == None:
            normalization = "in"
        else:
            normalization = "cin"

        # ------ Down branch -----------------------------
        H, W, D, k = H0, W0, D0, k0
        self.d1 = DownSample3D(
            x_dim=C0, filters=k, downsample=False, act_param=act_param
        )
        self.d2 = DenseBlock3D(
            x_shape=(k, D, H, W),
            act_param=act_param,
            out_channels=dense_int_out,
            layers=denselayers,
        )

        self.d3 = DownSample3D(x_dim=k, filters=2 * k, act_param=act_param)
        H, W, D, k = (H - 2) // 2 + 1, (W - 2) // 2 + 1, (D - 2) // 2 + 1, 2 * k
        self.d4 = DenseBlock3D(
            x_shape=(k, D, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=dense_int_out * 2,
            layers=denselayers,
        )

        self.d5 = DownSample3D(x_dim=k, filters=2 * k, act_param=act_param)
        H, W, D, k = (H - 2) // 2 + 1, (W - 2) // 2 + 1, (D - 2) // 2 + 1, 2 * k
        self.d6 = DenseBlock3D(
            x_shape=(k, H, D, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=dense_int_out * 4,
            layers=denselayers,
        )

        self.d7 = DownSample3D(x_dim=k, filters=2 * k, act_param=act_param)
        H, W, D, k = (H - 2) // 2 + 1, (W - 2) // 2 + 1, (D - 2) // 2 + 1, 2 * k
        self.d8 = DenseBlock3D(
            x_shape=(k, H, D, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=dense_int_out * 8,
            layers=denselayers,
        )
        # ------------------------------------------------

        # ----- Base of UNet------------------------------
        self.base = DenseBlock3D(
            x_shape=(k, H, D, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=dense_int_out * 8,
            layers=denselayers,
        )
        # -------------------------------------------------

        # ------ Up branch -----------------------------
        self.u1 = UpSample3D(x_dim=k, filters=k // 2, act_param=act_param)
        H, W, D, k = 2 * H, 2 * W, 2 * D, k // 2
        self.u2 = DenseBlock3D(
            x_shape=(k, D, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=dense_int_out * 4,
            layers=denselayers,
        )

        self.u3 = UpSample3D(
            x_dim=k, filters=k // 2, concat=True, old_x_dim=k, act_param=act_param
        )
        H, W, D, k = 2 * H, 2 * W, 2 * D, k // 2
        self.u4 = DenseBlock3D(
            x_shape=(k, D, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=dense_int_out * 2,
            layers=denselayers,
        )

        self.u5 = UpSample3D(
            x_dim=k, filters=k // 2, concat=True, old_x_dim=k, act_param=act_param
        )

        H, W, D, k = 2 * H, 2 * W, 2 * D, k // 2
        self.u6 = DenseBlock3D(
            x_shape=(k, D, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=dense_int_out,
            layers=denselayers,
        )

        self.u7 = UpSample3D(
            x_dim=k,
            filters=k,
            concat=True,
            old_x_dim=k,
            upsample=False,
            act_param=act_param,
        )

        self.u8 = DenseBlock3D(
            x_shape=(k, D, H, W),
            z_dim=z_dim,
            normalization=normalization,
            act_param=act_param,
            out_channels=dense_int_out,
            layers=denselayers,
        )

        self.u9 = UpSample3D(
            x_dim=k,
            filters=out_channels,
            upsample=False,
            act_param=act_param,
        )

        # Becoming 2D needs to be done in the
        self.u10 = UpSample2D(
            x_dim=D * out_channels,
            filters=out_channels,
            upsample=False,
            activation=False,
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

        x10_size = x10.size()
        x11 = x10.view(x10_size[0], x10_size[1] * x10_size[2], x10_size[3], x10_size[4])
        x11 = self.u10(input_x=x11)

        output = x11

        return output


class critic_dense_separate_2D(nn.Module):
    """Critic model using Denseblocks
    input_x and input_y are both assumed to have
    the shape (N, C, H, W)
    """

    def __init__(
        self,
        x_shape,
        y_shape,
        k0_x=24,
        k0_y=24,
        act_param=0.1,
        denselayers_x=3,
        denselayers_y=3,
        dense_int_out_x=16,
        dense_int_out_y=16,
        ds_k_x=4,
        ds_s_x=4,
        ds_k_y=4,
        ds_s_y=4,
    ):
        """
        x_shape does not include the number of samples N.
        """
        super(critic_dense_separate_2D, self).__init__()
        C0_x, H0x, W0x = x_shape
        C0_y, H0y, W0y = y_shape
        # ------ Convolution layers X -----------------------------
        H, W = H0x, W0x
        self.xcnn1 = DownSample2D(
            x_dim=C0_x, filters=k0_x, downsample=False, act_param=act_param
        )

        self.xcnn2 = DenseBlock2D(
            x_shape=(k0_x, H, W),
            act_param=act_param,
            normalization="ln",
            out_channels=dense_int_out_x,
            layers=denselayers_x,
        )

        self.xcnn3 = DownSample2D(
            x_dim=k0_x, filters=2 * k0_x, act_param=act_param, ds_k=ds_k_x, ds_s=ds_s_x
        )
        H, W = (H - 2) // 2 + 1, (W - 2) // 2 + 1
        self.xcnn4 = DenseBlock2D(
            x_shape=(2 * k0_x, H, W),
            act_param=act_param,
            normalization="ln",
            out_channels=2 * dense_int_out_x,
            layers=denselayers_x,
        )

        self.xcnn5 = DownSample2D(
            x_dim=2 * k0_x,
            filters=4 * k0_x,
            act_param=act_param,
            ds_k=ds_k_x,
            ds_s=ds_s_x,
        )
        H, W = (H - 2) // 2 + 1, (W - 2) // 2 + 1

        self.xcnn6 = DenseBlock2D(
            x_shape=(4 * k0_x, H, W),
            act_param=act_param,
            normalization="ln",
            out_channels=4 * dense_int_out_x,
            layers=denselayers_x,
        )

        self.xcnn7 = DownSample2D(x_dim=4 * k0_x, filters=8 * k0_x, act_param=act_param)
        H, W = (H - 2) // 2 + 1, (W - 2) // 2 + 1
        self.xcnn8 = DenseBlock2D(
            x_shape=(8 * k0_x, H, W),
            act_param=act_param,
            normalization="ln",
            out_channels=8 * dense_int_out_x,
            layers=denselayers_x,
        )
        Hx, Wx = H, W
        # ------ Convolution layers Y -----------------------------
        H, W = H0y, W0y
        self.ycnn1 = DownSample2D(
            x_dim=C0_y, filters=k0_y, downsample=False, act_param=act_param
        )

        self.ycnn2 = DenseBlock2D(
            x_shape=(k0_y, H, W),
            act_param=act_param,
            normalization="ln",
            out_channels=dense_int_out_y,
            layers=denselayers_y,
        )

        self.ycnn3 = DownSample2D(
            x_dim=k0_y, filters=2 * k0_y, act_param=act_param, ds_k=ds_k_y, ds_s=ds_s_y
        )
        H, W = (H - 2) // 2 + 1, (W - 2) // 2 + 1
        self.ycnn4 = DenseBlock2D(
            x_shape=(2 * k0_y, H, W),
            act_param=act_param,
            normalization="ln",
            out_channels=2 * dense_int_out_y,
            layers=denselayers_y,
        )

        self.ycnn5 = DownSample2D(
            x_dim=2 * k0_y,
            filters=4 * k0_y,
            act_param=act_param,
            ds_k=ds_k_y,
            ds_s=ds_s_y,
        )
        H, W = (H - 2) // 2 + 1, (W - 2) // 2 + 1

        self.ycnn6 = DenseBlock2D(
            x_shape=(4 * k0_y, H, W),
            act_param=act_param,
            normalization="ln",
            out_channels=4 * dense_int_out_y,
            layers=denselayers_y,
        )

        self.ycnn7 = DownSample2D(x_dim=4 * k0_y, filters=8 * k0_y, act_param=act_param)
        H, W = (H - 2) // 2 + 1, (W - 2) // 2 + 1
        self.ycnn8 = DenseBlock2D(
            x_shape=(8 * k0_y, H, W),
            act_param=act_param,
            normalization="ln",
            out_channels=8 * dense_int_out_y,
            layers=denselayers_y,
        )
        Hy, Wy = H, W
        # ----- Dense layers------------------------------
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(
            in_features=8 * k0_y * Hy * Wy + 8 * k0_x * Hx * Wx, out_features=128
        )
        self.LReLU = nn.ELU(alpha=act_param)
        self.LN = ApplyNormalization2D(x_shape=(128), normalization="ln")
        self.lin2 = nn.Linear(in_features=128, out_features=128)
        self.lin3 = nn.Linear(in_features=128, out_features=1)

        # ------------------------------------------------

    def forward(self, input_x, input_y):

        x = self.xcnn1(input_x=input_x)
        x = self.xcnn2(input_x=x)
        x = self.xcnn3(input_x=x)
        x = self.xcnn4(input_x=x)
        x = self.xcnn5(input_x=x)
        x = self.xcnn6(input_x=x)
        x = self.xcnn7(input_x=x)
        x = self.xcnn8(input_x=x)

        y = self.ycnn1(input_x=input_y)
        y = self.ycnn2(input_x=y)
        y = self.ycnn3(input_x=y)
        y = self.ycnn4(input_x=y)
        y = self.ycnn5(input_x=y)
        y = self.ycnn6(input_x=y)
        y = self.ycnn7(input_x=y)
        y = self.ycnn8(input_x=y)

        x = self.flat(x)
        y = self.flat(y)
        xy = cat((x, y), dim=1)

        xy = self.lin1(xy)
        xy = self.LReLU(xy)
        xy = self.LN(xy)
        xy = self.lin2(xy)
        xy = self.LReLU(xy)
        xy = self.LN(xy)
        xy = self.lin2(xy)
        xy = self.LReLU(xy)
        xy = self.LN(xy)
        output = self.lin3(xy)

        return output


class critic_dense_2D(nn.Module):
    """Critic model using Denseblocks
    input_x and input_y are both assumed to have
    the shape (N, C, H, W)
    """

    def __init__(
        self, x_shape, y_shape, k0=24, act_param=0.1, denselayers=3, dense_int_out=16
    ):
        """
        x_shape does not include the number of samples N.
        """
        super(critic_dense_2D, self).__init__()
        C0_x, H0, W0 = x_shape
        C0_y, H0, W0 = y_shape

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


class critic_dense_3D_and_2D(nn.Module):
    """Critic model using Denseblocks
    input_x and input_y are both assumed to have
    the shape (N, C, H, W)
    """

    def __init__(
        self,
        x_shape,
        y_shape,
        k0_x=24,
        k0_y=24,
        act_param=0.1,
        denselayers_x=3,
        denselayers_y=3,
        dense_int_out_x=16,
        dense_int_out_y=16,
    ):
        """
        x_shape does not include the number of samples N.
        """
        super(critic_dense_3D_and_2D, self).__init__()
        C0_x, xH0, xW0 = x_shape
        C0_y, yD0, yH0, yW0 = y_shape

        # ------ Convolution layers for the X (2D) ---------------------
        xH, xW = xH0, xW0
        self.xcnn1 = DownSample2D(
            x_dim=C0_x, filters=k0_x, downsample=False, act_param=act_param
        )

        self.xcnn2 = DenseBlock2D(
            x_shape=(k0_x, xH, xW),
            act_param=act_param,
            normalization="ln",
            out_channels=dense_int_out_x,
            layers=denselayers_x,
        )

        self.xcnn3 = DownSample2D(
            x_dim=k0_x, filters=2 * k0_x, act_param=act_param, ds_k=4, ds_s=4
        )
        xH, xW = (xH - 2) // 4 + 1, (xW - 2) // 4 + 1
        self.xcnn4 = DenseBlock2D(
            x_shape=(2 * k0_x, xH, xW),
            act_param=act_param,
            normalization="ln",
            out_channels=2 * dense_int_out_x,
            layers=denselayers_x,
        )

        self.xcnn5 = DownSample2D(
            x_dim=2 * k0_x, filters=4 * k0_x, act_param=act_param, ds_k=4, ds_s=4
        )
        xH, xW = (xH - 2) // 4 + 1, (xW - 2) // 4 + 1

        self.xcnn6 = DenseBlock2D(
            x_shape=(4 * k0_x, xH, xW),
            act_param=act_param,
            normalization="ln",
            out_channels=4 * dense_int_out_x,
            layers=denselayers_x,
        )

        self.xcnn7 = DownSample2D(x_dim=4 * k0_x, filters=8 * k0_x, act_param=act_param)
        xH, xW = (xH - 2) // 2 + 1, (xW - 2) // 2 + 1
        self.xcnn8 = DenseBlock2D(
            x_shape=(8 * k0_x, xH, xW),
            act_param=act_param,
            normalization="ln",
            out_channels=8 * dense_int_out_x,
            layers=denselayers_x,
        )

        # ------ Convolution layers for the Y (3D) -----------------------------
        yH, yW, yD = yH0, yW0, yD0
        self.ycnn1 = DownSample3D(
            x_dim=C0_y, filters=k0_y, downsample=False, act_param=act_param
        )

        self.ycnn2 = DenseBlock3D(
            x_shape=(k0_y, yD, yH, yW),
            act_param=act_param,
            normalization="ln",
            out_channels=dense_int_out_y,
            layers=denselayers_y,
            norm_elementwise_affine=False,
        )

        self.ycnn3 = DownSample3D(
            x_dim=k0_y, filters=2 * k0_y, act_param=act_param, ds_k=2, ds_s=2
        )
        yH, yW, yD = (yH - 2) // 2 + 1, (yW - 2) // 2 + 1, (yD - 2) // 2 + 1
        self.ycnn4 = DenseBlock3D(
            x_shape=(2 * k0_y, yD, yH, yW),
            act_param=act_param,
            normalization="ln",
            out_channels=2 * dense_int_out_y,
            layers=denselayers_y,
        )

        self.ycnn5 = DownSample3D(
            x_dim=2 * k0_y, filters=4 * k0_y, act_param=act_param, ds_k=2, ds_s=2
        )
        yH, yW, yD = (yH - 2) // 2 + 1, (yW - 2) // 2 + 1, (yD - 2) // 2 + 1

        self.ycnn6 = DenseBlock3D(
            x_shape=(4 * k0_y, yD, yH, yW),
            act_param=act_param,
            normalization="ln",
            out_channels=4 * dense_int_out_y,
            layers=denselayers_y,
        )

        self.ycnn7 = DownSample3D(x_dim=4 * k0_y, filters=8 * k0_y, act_param=act_param)
        yH, yW, yD = (yH - 2) // 2 + 1, (yW - 2) // 2 + 1, (yD - 2) // 2 + 1
        self.ycnn8 = DenseBlock3D(
            x_shape=(8 * k0_y, yD, yH, yW),
            act_param=act_param,
            normalization="ln",
            out_channels=8 * dense_int_out_y,
            layers=denselayers_y,
        )

        self.ycnn9 = DownSample3D(
            x_dim=8 * k0_y, filters=16 * k0_y, act_param=act_param
        )
        yH, yW, yD = (yH - 2) // 2 + 1, (yW - 2) // 2 + 1, (yD - 2) // 2 + 1

        # ----- Dense layers------------------------------
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(
            in_features=16 * k0_y * yD * yH * yW + 8 * k0_x * xW * xH, out_features=128
        )
        self.LReLU = nn.ELU(alpha=act_param)
        self.LN1 = ApplyNormalization2D(x_shape=(128), normalization="ln")
        self.lin2 = nn.Linear(in_features=128, out_features=128)
        self.lin3 = nn.Linear(in_features=128, out_features=32)
        self.LN2 = ApplyNormalization2D(x_shape=(32), normalization="ln")
        self.lin4 = nn.Linear(in_features=32, out_features=1)

        # ------------------------------------------------

    def forward(self, input_x, input_y):
        x = self.xcnn1(input_x=input_x)
        x = self.xcnn2(input_x=x)
        x = self.xcnn3(input_x=x)
        x = self.xcnn4(input_x=x)
        x = self.xcnn5(input_x=x)
        x = self.xcnn6(input_x=x)
        x = self.xcnn7(input_x=x)
        x = self.xcnn8(input_x=x)
        x = self.flat(x)

        y = self.ycnn1(input_x=input_y)
        y = self.ycnn2(input_x=y)
        y = self.ycnn3(input_x=y)
        y = self.ycnn4(input_x=y)
        y = self.ycnn5(input_x=y)
        y = self.ycnn6(input_x=y)
        y = self.ycnn7(input_x=y)
        y = self.ycnn8(input_x=y)
        y = self.ycnn9(input_x=y)
        y = self.flat(y)

        xy = cat((x, y), dim=1)

        xy = self.lin1(xy)
        xy = self.LReLU(xy)
        xy = self.LN1(xy)
        xy = self.lin2(xy)
        xy = self.LReLU(xy)
        xy = self.LN1(xy)
        xy = self.lin3(xy)
        xy = self.LReLU(xy)
        xy = self.LN2(xy)
        output = self.lin4(xy)

        return output
