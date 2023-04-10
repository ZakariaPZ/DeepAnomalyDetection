import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *

class DenseAE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 n_layers_encoder: int,
                 n_layers_decoder: int,
                 encoder_width: int,
                 decoder_width: int,
                 scaling_factor: int = 1/2.,
                 norm: Union[str, None] = None,
                 dropout: Union[float, None] = None,
                 type: str = 'dense',
                ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.encoder_width = encoder_width
        self.decoder_width = decoder_width
        self.scaling_factor = scaling_factor

        self.block = MLPBlock if type == 'dense' else ConvBlock

        # check that the scaling factor is valid
        if not (0 < scaling_factor < 1):
            raise ValueError(f'Invalid scaling factor: {scaling_factor}')
        
        # check if the number of layers, width and scaling factor are compatabile
        if encoder_width * scaling_factor ** (n_layers_encoder - 1) % 1 != 0:
            raise ValueError(f'Invalid combination of encoder params: {encoder_width * scaling_factor ** (n_layers_encoder - 1)}')
        if decoder_width * (1/scaling_factor) ** (n_layers_decoder - 1) % 1 != 0:
            raise ValueError(f'Invalid combination of decoder params: {decoder_width * (1/scaling_factor) ** (n_layers_decoder - 1)}')
        

        self.encoder = nn.Sequential(
            *[
                self.block(
                    input_dim if i == 0 else int(encoder_width * scaling_factor ** (i - 1)),
                    int(encoder_width * scaling_factor ** i),
                    norm=norm, dropout=dropout
                )
                for i in range(n_layers_encoder)
            ],
            nn.Linear(int(encoder_width * scaling_factor ** (n_layers_encoder - 1)), latent_dim)
        )

        self.decoder = nn.Sequential(
            *[
                self.block(
                    latent_dim if i == 0 else int(decoder_width * scaling_factor ** (n_layers_decoder - i)),
                    int(decoder_width * scaling_factor ** (n_layers_decoder - i - 1)),
                    norm=norm, dropout=dropout
                )
                for i in range(n_layers_decoder)
            ],
            nn.Linear(decoder_width, input_dim)
        )



    def forward(self, input):
        latent_repr = self.encode(input)
        output = self.decode(latent_repr)
        return output
        
    def encode(self, input):
        return self.encoder(input)

    def decode(self, latent_repr):
        return self.decoder(latent_repr)
    

class MLPBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
                 norm: Union[str, None] = None,
                 dropout: Union[float, None] = None
                ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation

        if norm == 'batch':
            self.norm = nn.BatchNorm1d(output_dim)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(output_dim)
        elif norm is None:
            self.norm = None
        else:
            raise ValueError(f'Invalid norm: {norm}')
        
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, input):
        output = self.linear(input)
        output = self.activation(output)
        if self.norm:
            output = self.norm(output)
        if self.dropout:
            output = self.dropout(output)
        return output
    
class ConvBlock(nn.Module):
    ...

if __name__ == "__main__":
    args ={
        'input_dim' : [28*28] * 12,
        "n_layers_encoder" : [2,2,2,1,1,1,2,2,2,4,4,4,],
        'encoder_width' : [392,784,1568,784,784,784,784,784,784,784,784,784],
        'n_layers_decoder' : [2,2,2,1,2,4,1,2,4,1,2,4,],
        'decoder_width' : [392,784,1568,784,784,784,784,784,784,784,784,784],
        'latent_dim' : [32] * 12,
        'scaling_factor' : [1/2] * 12,
        'norm' : ['batch'] * 12,
        'dropout' : [0.3] * 12,
    }

    for i in range(12):
        args_ = {k: v[i] for k, v in args.items()}
        print(args_)
        model = DenseAE(**args_)
        input = torch.randn(2, args_['input_dim'])
        output = model(input)
        assert output.shape == input.shape, "Autoencoder output shape does not match input shape"
        