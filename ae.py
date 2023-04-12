import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from torch.utils import data
from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer

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
    


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
    
        return x.view(self.shape)

class ConvAE(nn.Module):
    def __init__(self,
                 input_channels: int,
                 Height : int,
                 latent_dim: int,
                 n_layers_encoder: int,
                 n_layers_decoder: int,
                 encoder_width: int,
                 decoder_width: int,
                 scaling_factor: int = 1/2.,
                 norm: Union[str, None] = None,
                 dropout: Union[float, None] = None,
                 padding : int = None,
                 kernel_size : int = None,
                 pool_kernel : int = None
                ):
        super().__init__()

        self.input_dim = input_channels
        self.latent_dim = latent_dim
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.encoder_width = encoder_width
        self.decoder_width = decoder_width
        self.scaling_factor = scaling_factor 
        self.norm = norm
        self.dropout = dropout
        self.padding = padding
        self.kernel_size = kernel_size
        self.pool_kernel = int(pool_kernel) # convert kernel to int if float is passed
        self.block = ConvBlock
        self.Height = Height

        # check that the scaling factor is valid
        if not (0 < scaling_factor < 1):
            raise ValueError(f'Invalid scaling factor: {scaling_factor}')
        
        # check if the number of layers, width and scaling factor are compatabile
        if encoder_width * scaling_factor ** (n_layers_encoder - 1) % 1 != 0:
            raise ValueError(f'Invalid combination of encoder params: {encoder_width * scaling_factor ** (n_layers_encoder - 1)}')
        if decoder_width * (1/scaling_factor) ** (n_layers_decoder - 1) % 1 != 0:
            raise ValueError(f'Invalid combination of decoder params: {decoder_width * (1/scaling_factor) ** (n_layers_decoder - 1)}')
        
        # Encoder
        blocks = []
        for i in range(self.n_layers_encoder):
            in_channels = input_channels if i == 0 else int(encoder_width * scaling_factor ** (self.n_layers_encoder - i))
            out_channels = int(encoder_width * scaling_factor ** (self.n_layers_encoder - (i + 1)))
            blocks.append(
                self.block(
                    in_channels,
                    out_channels,
                    norm=norm, dropout=dropout,
                    conv_type='downsample'
                )
            )
     
        blocks.append(nn.Flatten())
        num_channels = self.encoder_width
        dims = int(Height*(1/self.pool_kernel)**(self.n_layers_encoder))
        blocks.append(nn.Linear(num_channels * dims * dims, self.latent_dim))
        self.encoder = nn.Sequential(*blocks)

        # Decoder
        decoder_dim = round(self.Height/2**self.n_layers_decoder)
        blocks_dec = [nn.Linear(self.latent_dim, num_channels * decoder_dim * decoder_dim), 
                  Reshape(-1, num_channels, decoder_dim, decoder_dim)] 
        for i in range(self.n_layers_decoder):
            H_in = round(Height*(1/self.pool_kernel)**(self.n_layers_decoder - i))
            H_out = round(Height*(1/self.pool_kernel)**(self.n_layers_decoder - (i+1)))
            stride = 2 # 2 for upsampling with conv_transpose
            padding = self.padding
            dilation = 1
            kernel_size = self.kernel_size
            output_padding = H_out - ((H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1)

            in_channels = num_channels if i == 0 else int(decoder_width * scaling_factor ** (i))
            out_channels = input_channels if i == self.n_layers_decoder - 1 else int(decoder_width * scaling_factor ** (i + 1))

            blocks_dec.append(
                self.block(
                    in_channels,
                    out_channels,
                    norm=norm, dropout=dropout, output_padding=output_padding,
                    conv_type='upsample',
                    stride=stride
                )
            )

        self.decoder = nn.Sequential(*blocks_dec)

    def forward(self, input):
        z = self.encoder(input)
        x_hat = self.decoder(z)
        return x_hat
        

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
    def __init__(self, 
                 input_dim : int,
                 output_dim : int,
                 kernel_size : int = 3,
                 stride : int = 1,
                 padding : int = 1,
                 activation : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU,
                 conv_type : str = 'downsample',
                 output_padding : int = 0,
                 norm = None,
                 dropout = None) -> None:
        super().__init__()

        self.conv_type = conv_type
        self.output_padding = output_padding

        if conv_type == 'downsample':

            self.conv = nn.Sequential(
                nn.Conv2d(input_dim, 
                          output_dim, 
                          kernel_size=kernel_size, 
                          stride=stride,
                          padding=padding),
                activation(),
                nn.MaxPool2d(2)
            )

        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(input_dim,
                                output_dim,
                                kernel_size=kernel_size,
                                padding=1,
                                stride=2,
                                output_padding=output_padding),
                activation()
            )
 
        self.norm = nn.BatchNorm2d(output_dim) if norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):

        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x



class PretrainModule(pl.LightningModule):
    def __init__(self,
                 input_channels: int,
                 Height : int,
                 latent_dim: int,
                 n_layers_encoder: int,
                 n_layers_decoder: int,
                 encoder_width: int,
                 decoder_width: int,
                 scaling_factor: int = 1/2.,
                 norm: Union[str, None] = None,
                 dropout: Union[float, None] = None,
                 padding : int = None,
                 kernel_size : int = None,
                 pool_kernel : int = None
                ):
        super(PretrainModule, self).__init__()

        self.model = ConvAE(input_channels, Height, latent_dim,
                            n_layers_encoder, n_layers_decoder, 
                            encoder_width, decoder_width, scaling_factor,
                            norm, dropout, padding, 
                            kernel_size, pool_kernel)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
      
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def train_dataloader(self):
        mnist_data = datasets.MNIST(root='./data', train=True, download=True, 
                                    transform=transforms.ToTensor())

        batch_size = 64
        # dataloader = data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True, num_workers=8)
        dataloader = data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

        return dataloader
        
    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":
    
    MLP_args ={
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


    CNN_args ={
        'input_channels' : [1],
        'Height' : [28],
        "n_layers_encoder" : [2],
        'encoder_width' : [16], # max num of channels
        'n_layers_decoder' : [2],
        'decoder_width' : [16], # max num of channels
        'latent_dim' : [32], # FC 
        'scaling_factor' : [1/2],
        'norm' : ['batch'],
        'dropout' : [0.3],
        'padding' : [1],
        'kernel_size' : [3],
        'pool_kernel' : [2],
    }


    for i in range(len(CNN_args['input_channels'])):
        args_ = {k: v[i] for k, v in CNN_args.items()}
        print(args_)
        # model = DenseAE(**args_)
        trainer = Trainer(max_epochs=10, fast_dev_run=False)
        model = PretrainModule(**args_)

        # input = torch.randn(2, args_['input_dim'])
        input = torch.randn(2, 1, 28, 28)
        output = model(input)
        assert output.shape == input.shape, "Autoencoder output shape does not match input shape"
        
        trainer.fit(model)