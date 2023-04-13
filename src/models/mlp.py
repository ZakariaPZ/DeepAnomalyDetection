import typing as th
import torch
import dypy as dy


class LazyLinearBlock(torch.nn.Module):
    def __init__(
        self,
        # linear args
        out_features: int,
        bias: bool = True,
        # activation
        activation: th.Optional[str] = "torch.nn.ReLU",
        activation_args: th.Optional[dict] = None,
        # batch norm
        normalization: th.Optional[str] = "torch.nn.BatchNorm1d",
        normalization_args: th.Optional[dict] = None,
        # residual
        residual: bool = True,
        residual_factor: float = 1.0,
        # dropout
        dropout: th.Optional[float] = None,
        # general parameters
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
        flatten: bool = True,
    ):
        super().__init__()
        self.linear = torch.nn.LazyLinear(
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
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
        self.residual, self.residual_factor, self.flatten = residual, residual_factor, flatten

    def forward(self, inputs: torch.Tensor, flatten: th.Optional[bool] = None) -> torch.Tensor:
        flatten = flatten if flatten is not None else self.flatten
        outputs = inputs.reshape(inputs.shape[0], -1) if flatten else inputs
        outputs = self.linear(outputs)
        outputs = self.activation(outputs) if self.activation else outputs
        outputs = self.normalizaton(outputs) if self.normalizaton else outputs
        outputs = self.dropout(outputs) if self.dropout else outputs
        if self.residual and self.residual_factor and outputs.shape[-1] == inputs.shape[-1]:
            outputs = outputs + self.residual_factor * inputs
        return outputs


class LazyMLP(torch.nn.Sequential):
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
        layers: th.List[int],
        out_features: int,
        flatten: bool = True,
        **kwargs,
    ):
        super().__init__()
        for idx, features in enumerate((layers or []) + [out_features]):
            args = dict(kwargs)
            if idx == len(layers or []):
                args["activation"] = None
                args["normalization"] = None
            self.add_module(f"block_{idx}", LazyLinearBlock(out_features=features, **args))
        self.flatten = flatten

    def forward(self, inputs, flatten: bool = None):
        flatten = flatten if flatten is not None else self.flatten
        results = super().forward(inputs if not flatten else inputs.reshape(inputs.shape[0], -1))
        return results
