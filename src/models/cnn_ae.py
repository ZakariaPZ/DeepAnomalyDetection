from .cnn import LazyConv
import typing as th
import torch

class CNNAutoEncoder(torch.nn.Sequential):
    def __init__(
        self,
        input_shape: th.Union[th.List[int], th.Tuple[int, int, int]],
        encoder_depth: int,
        decoder_depth: int,
        encoder_width: int,
        decoder_width: int,
        latent_dim: int,
        scaling_factor: float = 0.5,
        stride: int = 2,
        padding: int = 1,
        kernel_size: int = 3,
        encoder_args: th.Optional[dict] = None,  # overrides args for encoder
        decoder_args: th.Optional[dict] = None,  # overrides args for decoder
        vae: bool = False,
        **args: th.Optional[dict],  # see LazyConv for args
    ):
        super().__init__()

        self.encoder_depth, self.decoder_depth = encoder_depth, decoder_depth
        self.encoder_width, self.decoder_width = encoder_width, decoder_width

        self.input_shape = input_shape if isinstance(input_shape, tuple) else tuple(input_shape)
        self.scaling_factor, self.latent_dim = scaling_factor, latent_dim

        self.vae = vae  # if true, do reparameterization trick

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

        # Encoder
        self.encoder = LazyConv(
            layers=[(int(encoder_width * scaling_factor ** (encoder_depth - (i + 1))), None) for i in range(encoder_depth)],
            conv_type='downsample',
            **encoder_args
        )
        encoder_dim = round(self.input_shape[1]*(1/stride)**(self.encoder_depth))
        self.encoder_linear = torch.nn.Linear(encoder_width * encoder_dim * encoder_dim, self.latent_dim)

        self.encoder.add_module('flatten', module=torch.nn.Flatten())
        self.encoder.add_module('encoder_linear', module=self.encoder_linear)

        # Decoder
        self.decoder_dim = round(self.input_shape[1]/2**self.decoder_depth)
        self.decoder_linear = torch.nn.Linear(self.latent_dim, self.decoder_width * self.decoder_dim * self.decoder_dim)

        decoder_layers = []
        for i in range(decoder_depth):
            H_in = round(self.input_shape[1]*(1/stride)**(decoder_depth - i))
            H_out = round(self.input_shape[1]*(1/stride)**(decoder_depth - (i+1)))
            dilation = 1
            output_padding = H_out - ((H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1)
            out_channels = self.input_shape[0] if i == decoder_depth - 1 else int(decoder_width * scaling_factor ** (i))

            decoder_layers.append((out_channels, output_padding))

        self.decoder = LazyConv(
            decoder_layers,
            conv_type='upsample',
            **decoder_args
        )


    def forward(
        self,
        inputs: th.Optional[torch.Tensor] = None,
        latent: th.Optional[torch.Tensor] = None,
        reparameterize: th.Optional[bool] = None,
    ):
        assert inputs is not None or latent is not None, "Must provide either inputs or latent"
        x = self.encoder(inputs) if latent is None else latent

        reparameterize = reparameterize if reparameterize is not None else self.vae
        if reparameterize:
            # split the latent vector into mean and log variance
            mu, log_variance = x.chunk(2, dim=1)
            # sample from the latent space
            x = mu + torch.exp(log_variance / 2) * torch.randn_like(mu)
        
        x = self.decoder_linear(x)
        x = x.view(-1, self.decoder_width, self.decoder_dim, self.decoder_dim)
        x = self.decoder(x)
        return x
