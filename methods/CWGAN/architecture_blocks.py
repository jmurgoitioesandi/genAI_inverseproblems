from torch import (
    cat,
    rsqrt,
    Tensor,
)
import torch.nn as nn


class CondInsNorm2D(nn.Module):
    """Implementing conditional instance normalization
    where input_x is normalized wrt input_z
    input_x is assumed to have the shape (N, x_dim, H, W)
    input_z is assumed to have the shape (N, z_dim, 1, 1)
    """

    def __init__(self, x_dim, z_dim, eps=1.0e-6, act_param=0.1):
        super(CondInsNorm2D, self).__init__()
        self.eps = eps
        self.z_shift = nn.Sequential(
            nn.Conv2d(in_channels=z_dim, out_channels=x_dim, kernel_size=1, stride=1),
            nn.ELU(alpha=act_param),
        )
        self.z_scale = nn.Sequential(
            nn.Conv2d(in_channels=z_dim, out_channels=x_dim, kernel_size=1, stride=1),
            nn.ELU(alpha=act_param),
        )

    def forward(self, input_x, input_z):
        x_size = input_x.size()
        assert len(x_size) == 4
        assert len(input_z.size()) == 4

        shift = self.z_shift(input_z)
        scale = self.z_scale(input_z)
        x_reshaped = input_x.view(x_size[0], x_size[1], x_size[2] * x_size[3])
        x_mean = x_reshaped.mean(2, keepdim=True)
        x_var = x_reshaped.var(2, keepdim=True)
        x_rstd = rsqrt(x_var + self.eps)  # reciprocal sqrt
        x_s = ((x_reshaped - x_mean) * x_rstd).view(*x_size)
        output = x_s * scale + shift
        return output


class CondInsNorm3D(nn.Module):
    """Implementing conditional instance normalization
    where input_x is normalized wrt input_z
    input_x is assumed to have the shape (N, x_dim, D, H, W)
    input_z is assumed to have the shape (N, z_dim, 1, 1, 1)
    """

    def __init__(self, x_dim, z_dim, eps=1.0e-6, act_param=0.1):
        super(CondInsNorm3D, self).__init__()
        self.eps = eps
        self.z_shift = nn.Sequential(
            nn.Conv3d(in_channels=z_dim, out_channels=x_dim, kernel_size=1, stride=1),
            nn.ELU(alpha=act_param),
        )
        self.z_scale = nn.Sequential(
            nn.Conv3d(in_channels=z_dim, out_channels=x_dim, kernel_size=1, stride=1),
            nn.ELU(alpha=act_param),
        )

    def forward(self, input_x, input_z):
        x_size = input_x.size()

        assert len(x_size) == 5
        assert len(input_z.size()) == 5

        shift = self.z_shift(input_z)
        scale = self.z_scale(input_z)
        x_reshaped = input_x.view(
            x_size[0], x_size[1], x_size[2] * x_size[3] * x_size[4]
        )
        x_mean = x_reshaped.mean(2, keepdim=True)
        x_var = x_reshaped.var(2, keepdim=True)
        x_rstd = rsqrt(x_var + self.eps)  # reciprocal sqrt
        x_s = ((x_reshaped - x_mean) * x_rstd).view(*x_size)
        output = x_s * scale + shift
        return output


class InsNorm2D(nn.Module):
    """Implementing conditional instance normalization
    where input_x is normalized along the feature direction
    This is different from the BatchNorm base implementation
    in Pytorch
    input_x is assumed to have the shape (N, x_dim, H, W)
    """

    def __init__(self, x_dim, eps=1.0e-6, act_param=0.1, affine=True):
        super(InsNorm2D, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(Tensor(x_dim))
        self.shift = nn.Parameter(Tensor(x_dim))
        self.affine = affine
        # initialize weights
        if self.affine:
            nn.init.normal_(self.shift)
            nn.init.normal_(self.scale)

    def forward(self, input_x):
        x_size = input_x.size()
        assert len(x_size) == 4
        x_reshaped = input_x.view(x_size[0], x_size[1], x_size[2] * x_size[3])
        x_mean = x_reshaped.mean(2, keepdim=True)
        x_var = x_reshaped.var(2, keepdim=True)
        x_rstd = rsqrt(x_var + self.eps)  # reciprocal sqrt
        x_s = ((x_reshaped - x_mean) * x_rstd).view(*x_size)

        if self.affine:
            output = x_s * self.scale[:, None, None] + self.shift[:, None, None]
        else:
            output = x_s
        return output


class InsNorm3D(nn.Module):
    """Implementing conditional instance normalization
    where input_x is normalized along the feature direction
    This is different from the BatchNorm base implementation
    in Pytorch
    input_x is assumed to have the shape (N, x_dim, D, H, W)
    """

    def __init__(self, x_dim, eps=1.0e-6, act_param=0.1, affine=True):
        super(InsNorm3D, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(Tensor(x_dim))
        self.shift = nn.Parameter(Tensor(x_dim))
        self.affine = affine
        # initialize weights
        if self.affine:
            nn.init.normal_(self.shift)
            nn.init.normal_(self.scale)

    def forward(self, input_x):
        x_size = input_x.size()
        assert len(x_size) == 5
        x_reshaped = input_x.view(
            x_size[0], x_size[1], x_size[2] * x_size[3] * x_size[3]
        )
        x_mean = x_reshaped.mean(2, keepdim=True)
        x_var = x_reshaped.var(2, keepdim=True)
        x_rstd = rsqrt(x_var + self.eps)  # reciprocal sqrt
        x_s = ((x_reshaped - x_mean) * x_rstd).view(*x_size)

        if self.affine:
            output = x_s * self.scale[:, None, None] + self.shift[:, None, None]
        else:
            output = x_s
        return output


class ApplyNormalization2D(nn.Module):
    """Normalizing input_x.
    input_x is assumed to have the shape (N, x_dim, H, W)
    If used, input_z is assumed to have the shape (N, z_dim, 1, 1)
    """

    def __init__(
        self, x_shape, z_dim=None, normalization=None, elementwise_affine=True
    ):
        """
        NOTE: x_shape does not include the number of samples N.
              Thus, x_shape[0] will give the channel size
        """
        super(ApplyNormalization2D, self).__init__()
        if normalization == "cin":
            assert z_dim is not None
            self.xnorm = CondInsNorm2D(x_shape[0], z_dim)
        elif normalization == "bn":
            self.xnorm = nn.BatchNorm2d(x_shape[0])
        elif normalization == "ln":
            self.xnorm = nn.LayerNorm(x_shape, elementwise_affine=elementwise_affine)
        elif normalization == "in":
            self.xnorm = InsNorm2D(x_shape[0], affine=True)
        else:
            self.xnorm = nn.Identity()

    def forward(self, input_x, input_z=None):
        if input_z is None:
            out = self.xnorm(input_x)
        else:
            out = self.xnorm(input_x, input_z)
        return out


class ApplyNormalization3D(nn.Module):
    """Normalizing input_x.
    input_x is assumed to have the shape (N, x_dim, D, H, W)
    If used, input_z is assumed to have the shape (N, z_dim, 1, 1, 1)
    """

    def __init__(
        self, x_shape, z_dim=None, normalization=None, elementwise_affine=True
    ):
        """
        NOTE: x_shape does not include the number of samples N.
              Thus, x_shape[0] will give the channel size
        """
        super(ApplyNormalization3D, self).__init__()
        if normalization == "cin":
            assert z_dim is not None
            self.xnorm = CondInsNorm3D(x_shape[0], z_dim)
        elif normalization == "bn":
            self.xnorm = nn.BatchNorm3d(x_shape[0])
        elif normalization == "ln":
            self.xnorm = nn.LayerNorm(x_shape, elementwise_affine=elementwise_affine)
        elif normalization == "in":
            self.xnorm = InsNorm3D(x_shape[0], affine=True)
        else:
            self.xnorm = nn.Identity()

    def forward(self, input_x, input_z=None):
        if input_z is None:
            out = self.xnorm(input_x)
        else:
            out = self.xnorm(input_x, input_z)
        return out


class ResBlock2D(nn.Module):
    """Implementing a single ResBlock
    input_x is assumed to have the shape (N, x_dim, H, W)
    If used, input_z is assumed to have the shape (N, z_dim, 1, 1)
    """

    def __init__(self, x_shape, z_dim=None, normalization=None, act_param=0.1):
        """
        x_shape does not include the number of samples N.
        """
        super(ResBlock2D, self).__init__()

        self.norm = ApplyNormalization2D(x_shape, z_dim, normalization)
        self.conv = nn.Conv2d(
            in_channels=x_shape[0], out_channels=x_shape[0], kernel_size=1, stride=1
        )
        self.branch = nn.ModuleList(
            self.build_branch(x_shape, z_dim, normalization, act_param)
        )
        # for i, layer in enumerate(self.branch):
        #     self.add_module(f"branch_layer_{i}", layer)

    def build_branch(self, x_shape, z_dim, normalization, act_param):
        """
        x_shape does not include the number of samples N.
        """
        model = [
            nn.ELU(alpha=act_param),
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=x_shape[0], out_channels=x_shape[0], kernel_size=3, stride=1
            ),
            ApplyNormalization2D(x_shape, z_dim, normalization),
            nn.ELU(alpha=act_param),
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=x_shape[0], out_channels=x_shape[0], kernel_size=3, stride=1
            ),
        ]
        return model

    def forward(self, input_x, input_z=None):
        x = self.norm(input_x, input_z)
        x1 = self.conv(x)
        # Hacky way of using variable number of inputs per layer
        for i, layer in enumerate(self.branch):
            if i == 3:  # This is the normalization layer
                x = layer(x, input_z)
            else:
                x = layer(x)
        output = x + x1
        return output


class DenseBlock2D(nn.Module):
    """Implementing a single DenseBlock (see Densely Connected Convolution Networks by Huang et al.)
    input_x is assumed to have the shape (N, x_dim, H, W)
    If used, input_z is assumed to have the shape (N, z_dim, 1, 1)
    """

    def __init__(
        self,
        x_shape,
        z_dim=None,
        normalization=None,
        act_param=0.1,
        out_channels=16,
        layers=4,
    ):
        """
        x_shape does not include the number of samples N.
        """
        super(DenseBlock2D, self).__init__()

        self.model = nn.ModuleList(
            self.build_block(
                x_shape=x_shape,
                z_dim=z_dim,
                normalization=normalization,
                act_param=act_param,
                out_channels=out_channels,
                layers=layers,
            )
        )

        self.layers = layers

    def build_block(
        self, x_shape, z_dim, normalization, act_param, out_channels, layers
    ):
        """
        x_shape does not include the number of samples N.
        """
        model = []

        for i in range(layers):
            x_shape_i = (
                x_shape[0] + i * out_channels,
                x_shape[1],
                x_shape[2],
            )
            model.append(ApplyNormalization2D(x_shape_i, z_dim, normalization))
            model.append(nn.ELU(alpha=act_param))
            model.append(nn.ReflectionPad2d(1))
            if i < layers - 1:
                model.append(
                    nn.Conv2d(
                        in_channels=x_shape_i[0],
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                    )
                )
            else:  # Last layer has same number of output channels as initial input
                model.append(
                    nn.Conv2d(
                        in_channels=x_shape_i[0],
                        out_channels=x_shape[0],
                        kernel_size=3,
                        stride=1,
                    )
                )

        return model

    def forward(self, input_x, input_z=None):
        # Hacky way of using variable number of inputs per layer
        for i in range(0, 4 * self.layers, 4):
            if i == 0:
                x_in = input_x
            else:
                x_in = cat((x_in, x), dim=1)
            x = self.model[i](x_in, input_z)
            x = self.model[i + 1](x)
            x = self.model[i + 2](x)
            x = self.model[i + 3](x)

        return x


class DenseBlock3D(nn.Module):
    """Implementing a single DenseBlock (see Densely Connected Convolution Networks by Huang et al.)
    input_x is assumed to have the shape (N, x_dim, D, H, W)
    If used, input_z is assumed to have the shape (N, z_dim, 1, 1)
    """

    def __init__(
        self,
        x_shape,
        z_dim=None,
        normalization=None,
        act_param=0.1,
        out_channels=16,
        layers=4,
        norm_elementwise_affine=True,
    ):
        """
        x_shape does not include the number of samples N.
        """
        super(DenseBlock3D, self).__init__()

        self.model = nn.ModuleList(
            self.build_block(
                x_shape=x_shape,
                z_dim=z_dim,
                normalization=normalization,
                act_param=act_param,
                out_channels=out_channels,
                layers=layers,
                norm_elementwise_affine=norm_elementwise_affine,
            )
        )

        self.layers = layers

    def build_block(
        self,
        x_shape,
        z_dim,
        normalization,
        act_param,
        out_channels,
        layers,
        norm_elementwise_affine=True,
    ):
        """
        x_shape does not include the number of samples N.
        """
        model = []

        for i in range(layers):
            x_shape_i = (
                x_shape[0] + i * out_channels,
                x_shape[1],
                x_shape[2],
                x_shape[3],
            )
            model.append(
                ApplyNormalization3D(
                    x_shape_i,
                    z_dim,
                    normalization,
                    elementwise_affine=norm_elementwise_affine,
                )
            )
            model.append(nn.ELU(alpha=act_param))
            model.append(nn.ReflectionPad3d(1))
            if i < layers - 1:
                model.append(
                    nn.Conv3d(
                        in_channels=x_shape_i[0],
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                    )
                )
            else:  # Last layer has same number of output channels as initial input
                model.append(
                    nn.Conv3d(
                        in_channels=x_shape_i[0],
                        out_channels=x_shape[0],
                        kernel_size=3,
                        stride=1,
                    )
                )

        return model

    def forward(self, input_x, input_z=None):
        # Hacky way of using variable number of inputs per layer
        for i in range(0, 4 * self.layers, 4):
            if i == 0:
                x_in = input_x
            else:
                x_in = cat((x_in, x), dim=1)
            x = self.model[i](x_in, input_z)
            x = self.model[i + 1](x)
            x = self.model[i + 2](x)
            x = self.model[i + 3](x)

        return x


class DownSample2D(nn.Module):
    """Implementing a downsampling using average pooling
    input_x is assumed to have the shape (N, x_dim, H, W)
    """

    def __init__(
        self,
        x_dim,
        filters,
        downsample=True,
        activation=True,
        act_param=0.1,
        ds_k=2,
        ds_s=2,
    ):
        super(DownSample2D, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(
            in_channels=x_dim, out_channels=filters, kernel_size=3, stride=1
        )
        self.LReLU = nn.ELU(alpha=act_param)
        self.pool = nn.AvgPool2d(kernel_size=ds_k, stride=ds_s)
        self.downsample = downsample
        self.activation = activation

    def forward(self, input_x):
        x = self.pad(input_x)
        x = self.conv(x)
        if self.activation:
            x = self.LReLU(x)
        if self.downsample:
            x = self.pool(x)
        return x


class DownSample3D(nn.Module):
    """Implementing a downsampling using average pooling
    input_x is assumed to have the shape (N, x_dim, D, H, W)
    """

    def __init__(
        self,
        x_dim,
        filters,
        downsample=True,
        activation=True,
        act_param=0.1,
        ds_k=2,
        ds_s=2,
    ):
        super(DownSample3D, self).__init__()
        self.pad = nn.ReflectionPad3d(1)
        self.conv = nn.Conv3d(
            in_channels=x_dim, out_channels=filters, kernel_size=3, stride=1
        )
        self.LReLU = nn.ELU(alpha=act_param)
        self.pool = nn.AvgPool3d(kernel_size=ds_k, stride=ds_s)
        self.downsample = downsample
        self.activation = activation

    def forward(self, input_x):
        x = self.pad(input_x)
        x = self.conv(x)
        if self.activation:
            x = self.LReLU(x)
        if self.downsample:
            x = self.pool(x)
        return x


class UpSample2D(nn.Module):
    """Implementing a upsampling with skip connection
    concatenations
    input_x is assumed to have the shape (N, x_dim, H, W)
    If used, old_x is assumed to have size (N, old_x_dim, H, W)
    where old_x_dim need not be the same as x_dim
    """

    def __init__(
        self,
        x_dim,
        filters,
        upsample=True,
        concat=False,
        old_x_dim=0,
        activation=True,
        act_param=0.1,
    ):
        super(UpSample2D, self).__init__()
        self.upsample = upsample
        self.activation = activation
        self.concat = concat
        self.filters = filters

        if self.concat:
            input_dim = x_dim + old_x_dim
        else:
            input_dim = x_dim

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(
            in_channels=input_dim, out_channels=filters, kernel_size=3, stride=1
        )
        self.LReLU = nn.ELU(alpha=act_param)
        self.rpool = nn.Upsample(scale_factor=2)

    def forward(self, input_x, old_x=None):
        if self.concat and old_x is not None:
            x = cat((input_x, old_x), dim=1)
        else:
            x = input_x
        x = self.pad(x)
        x = self.conv(x)
        if self.activation:
            x = self.LReLU(x)
        if self.upsample:
            x = self.rpool(x)
        return x


class UpSample3D(nn.Module):
    """Implementing a upsampling with skip connection
    concatenations
    input_x is assumed to have the shape (N, x_dim, D, H, W)
    If used, old_x is assumed to have size (N, old_x_dim, D, H, W)
    where old_x_dim need not be the same as x_dim
    """

    def __init__(
        self,
        x_dim,
        filters,
        upsample=True,
        concat=False,
        old_x_dim=0,
        activation=True,
        act_param=0.1,
    ):
        super(UpSample3D, self).__init__()
        self.upsample = upsample
        self.activation = activation
        self.concat = concat
        self.filters = filters

        if self.concat:
            input_dim = x_dim + old_x_dim
        else:
            input_dim = x_dim

        self.pad = nn.ReflectionPad3d(1)
        self.conv = nn.Conv3d(
            in_channels=input_dim, out_channels=filters, kernel_size=3, stride=1
        )
        self.LReLU = nn.ELU(alpha=act_param)
        self.rpool = nn.Upsample(scale_factor=2)

    def forward(self, input_x, old_x=None):
        if self.concat and old_x is not None:
            x = cat((input_x, old_x), dim=1)
        else:
            x = input_x
        x = self.pad(x)
        x = self.conv(x)
        if self.activation:
            x = self.LReLU(x)
        if self.upsample:
            x = self.rpool(x)
        return x
