from .mlp import LazyMLP
import typing as th
import torch


class MLPAutoEncoder(torch.nn.Sequential):
    def __init__(
        self,
        input_shape: th.Union[th.List[int], th.Tuple[int, int, int]],
        encoder_depth: int,
        decoder_depth: int,
        encoder_width: int,
        decoder_width: int,
        latent_dim: int,
        scaling_factor: float = 0.5,
        args: th.Optional[dict] = None,  # see LazyMLP for args
        encoder_args: th.Optional[dict] = None,  # overrides args
        decoder_args: th.Optional[dict] = None,  # overrides args
    ):
        super().__init__()

        self.encoder_depth, self.decoder_depth = encoder_depth, decoder_depth
        self.encoder_width, self.decoder_width = encoder_width, decoder_width

        self.input_shape = input_shape
        self.scaling_factor, self.latent_dim = scaling_factor, latent_dim

        # check that the scaling factor is valid
        if not (0 < scaling_factor < 1):
            raise ValueError(f"Invalid scaling factor: {scaling_factor}, must be between 0 and 1")

        # check if the number of layers, width and scaling factor are compatabile
        if encoder_width * scaling_factor ** (decoder_depth) % 1 != 0:
            raise ValueError(
                f"Invalid combination of encoder params, encoder_width * scaling_factor ** \
                (encoder_width) must be a positive integer: {encoder_width * scaling_factor ** (encoder_width)}"
            )
        if decoder_width * (1 / scaling_factor) ** (decoder_depth) % 1 != 0:
            raise ValueError(
                f"Invalid combination of encoder params, encoder_width * scaling_factor ** \
                (encoder_width) must be a positive integer: {decoder_width * scaling_factor ** (encoder_width)}"
            )

        args = {} if args is None else args
        encoder_args = args if encoder_args is None else {**args, **encoder_args}
        decoder_args = args if decoder_args is None else {**args, **decoder_args}

        self.encoder = LazyMLP(
            out_features=latent_dim,
            layers=[int(encoder_width * scaling_factor**i) for i in range(encoder_depth)],
            **encoder_args,
        )

        self.decoder = LazyMLP(
            out_features=input_shape[0] * input_shape[1] * input_shape[2],
            layers=[int(decoder_width * scaling_factor**i) for i in range(decoder_depth)][::-1],
            **decoder_args,
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(-1, *self.input_shape)
