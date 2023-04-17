import typing as th
import torch
import dypy as dy


class LazyConvBlock(torch.nn.Module):
    def __init__(
        self,
        # cnn args
        out_features: int,
        bias: bool = True,
        kernel_size : int = 3,
        stride : int = 2,
        padding : int = 1,
        output_padding : int = 0,
        # activation
        activation: th.Optional[str] = "torch.nn.LeakyReLU",
        activation_args: th.Optional[dict] = None,
        # batch norm
        normalization: th.Optional[str] = "torch.nn.BatchNorm2d",
        normalization_args: th.Optional[dict] = None,
        # Upsampling or downsampling
        conv_type: str = 'downsample',
        # dropout
        dropout: th.Optional[float] = None,
        # general parameters
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None
    ):
        super().__init__()

        if conv_type == 'downsample':

            self.conv = torch.nn.LazyConv2d(
                        out_channels=out_features, 
                        kernel_size=kernel_size, 
                        stride=stride,
                        padding=padding,
                        bias=bias,
                        device=device,
                        dtype=dtype
                        )
        else:
            self.conv = torch.nn.LazyConvTranspose2d(
                        out_channels=out_features, 
                        kernel_size=kernel_size, 
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                        bias=bias,
                        device=device,
                        dtype=dtype
                        )

        self.activation = dy.get_value(activation)(**(activation_args or dict())) if activation else None
        self.normalizaton = ( 
            dy.functions.call_with_dynamic_args(
                dy.get_value(normalization),  #
                num_features=out_features,
                dtype=dtype,
                device=device,
                **(normalization_args or dict()),
            )
            if normalization is not None
            else None
        )
        self.dropout = torch.nn.Dropout(dropout) if dropout else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv(inputs)
        outputs = self.activation(outputs) if self.activation else outputs
        outputs = self.normalizaton(outputs) if self.normalizaton else outputs
        outputs = self.dropout(outputs) if self.dropout else outputs

        return outputs


class LazyConv(torch.nn.Sequential):
    """
    Multi-layer Perceptron (MLP) Network.
    Attributes:
        in_features: The number of input features.
        layers: Hidden layers of the MLP. (list of number of features)
        bias: If True, adds a bias term to linear computations.
        residual: If True, adds a residual connections.
        residual_factor: factor to scale the residual connection by (defaults to 1.0).
        activation: The activation function to use (None for linear model).
        activation_args: The arguments to pass to the activation function.
        batch_norm: If True, adds a batch normalization layer in blocks.
        batch_norm_args: The arguments to pass to the batch normalization layers.
        device: The device to use.
        dtype: The data type to use.
    """

    def __init__(
        self,
        layers: th.List[th.Tuple[int, int]],
        conv_type: str,
        **kwargs
    ):
        super().__init__()
        for idx, items in enumerate(layers or []):
            features, output_padding = items
            args = dict(kwargs)
            if idx == len(layers or []):
                args["activation"] = None
                args["normalization"] = None
            
            self.add_module(f"block_{idx}", LazyConvBlock(out_features=features, 
                                                          output_padding=output_padding,
                                                          conv_type=conv_type,
                                                           **args))