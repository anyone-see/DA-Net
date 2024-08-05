import torch.nn as nn
import numpy as np


def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride=stride, padding=paddings)]
    if (batch_norm):
        model.append(nn.BatchNorm2d(output_channels))
    if (Relu):
        model.append(nn.ReLU())
    return nn.Sequential(*model)


class EchoNet(nn.Module):
    ## strucure adapted from VisualEchoes [ECCV 2020]
    r"""A Simple 3-Conv CNN followed by a fully connected layer
    """

    def __init__(self, conv1x1_dim, audio_shape, audio_feature_length, repeat=4):
        super(EchoNet, self).__init__()
        self.repeat = repeat
        self._n_input_audio = audio_shape[0]
        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]
        cnn_dims = np.array(audio_shape[1:], dtype=np.float32)

        for kernel_size, stride in zip(
                self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.conv1 = create_conv(self._n_input_audio, 32, kernel=self._cnn_layers_kernel_size[0], paddings=0,
                                 stride=self._cnn_layers_stride[0])
        self.conv2 = create_conv(32, 64, kernel=self._cnn_layers_kernel_size[1], paddings=0,
                                 stride=self._cnn_layers_stride[1])
        self.conv3 = create_conv(64, conv1x1_dim, kernel=self._cnn_layers_kernel_size[2], paddings=0,
                                 stride=self._cnn_layers_stride[2])
        layers = [self.conv1, self.conv2, self.conv3]
        self.feature_extraction = nn.Sequential(*layers)
        self.conv1x1 = create_conv(conv1x1_dim * cnn_dims[0] * cnn_dims[1], audio_feature_length, 1, 0)

    def _conv_output_dim(
            self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                                (
                                        dimension[i]
                                        + 2 * padding[i]
                                        - dilation[i] * (kernel_size[i] - 1)
                                        - 1
                                )
                                / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.shape[0], -1, 1, 1)
        x = self.conv1x1(x)
        x = x.repeat(1, 1, self.repeat, self.repeat)
        return [0, 0, 0, x]
