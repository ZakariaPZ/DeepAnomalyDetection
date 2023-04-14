from typing import *

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy, confusion_matrix, auroc
from torchvision import transforms, datasets
from torch.utils import data

cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")

class DenseAE(pl.LightningModule):
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
                 lr: float = 1e-3,
                ):
        super().__init__()

        self.threshold = 0

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.encoder_width = encoder_width
        self.decoder_width = decoder_width
        self.scaling_factor = scaling_factor
        self.lr = lr
        self.block = MLPBlock

        # check that the scaling factor is valid
        if not (0 < scaling_factor < 1):
            raise ValueError(f'Invalid scaling factor: {scaling_factor}')
        
        # check if the number of layers, width and scaling factor are compatabile
        if encoder_width * scaling_factor ** (n_layers_encoder) % 1 != 0:
            raise ValueError(f'Invalid combination of encoder params: {encoder_width * scaling_factor ** (n_layers_encoder - 1)}')
        if decoder_width * (1/scaling_factor) ** (n_layers_decoder) % 1 != 0:
            raise ValueError(f'Invalid combination of decoder params: {decoder_width * (1/scaling_factor) ** (n_layers_decoder - 1)}')
        

        self.encoder = nn.Sequential(
            *[
                self.block(
                    input_dim if i == 0 else int(encoder_width * scaling_factor ** (i - 1)),
                    int(encoder_width * scaling_factor ** i),
                    norm=norm, dropout=dropout
                )
                for i in range(n_layers_encoder + 1)
            ],
            nn.Linear(int(encoder_width * scaling_factor ** (n_layers_encoder)), latent_dim)
        )

        self.decoder = nn.Sequential(
            *[
                self.block(
                    latent_dim if i == 0 else int(decoder_width * scaling_factor ** (n_layers_decoder - i + 1)),
                    int(decoder_width * scaling_factor ** (n_layers_decoder - i)),
                    norm=norm, dropout=dropout
                )
                for i in range(n_layers_decoder + 1)
            ],
            nn.Linear(decoder_width, input_dim)
        )



    def forward(self, input):
        input = input.reshape(input.shape[0], -1)
        latent_repr = self.encode(input)
        output = self.decode(latent_repr)
        return output
        
    def encode(self, input):
        return self.encoder(input)

    def decode(self, latent_repr):
        return self.decoder(latent_repr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x.reshape(x.shape[0], -1))
        self.log('train_loss', loss)
        self.threshold = F.mse_loss(x_hat, x.reshape(x.shape[0], -1), reduction = 'mean')
        self.log('threshold', self.threshold)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        # # get indexes of class we train on
        # normal_class_idx = torch.where(y == self.normal_class)[0]
                
        # x_hat = self(x)
        # # get loss of model only for the class that we trained on
        # loss = F.mse_loss(
        #     x_hat[normal_class_idx],
        #     x[normal_class_idx]\
        #         .reshape(normal_class_idx.shape[0], -1)
        # )
        # # get reconstruction error for every example
        # all_mse = F.mse_loss(x_hat, x.reshape(x.shape[0], -1), reduction='none').mean(dim=-1)

        # # get classification based on threshold
        # y_hat = torch.where(all_mse > self.threshold, torch.ones_like(y), torch.zeros_like(y))
        
        # # get anomaly accuracy
        # acc = accuracy(
        #     y_hat,
        #     torch.where(y == self.normal_class, torch.zeros_like(y), torch.ones_like(y)),
        #     task='binary'
        # )

        x_hat = self(x)
        loss = F.mse_loss(x_hat, x.reshape(x.shape[0], -1))

        all_mse = F.mse_loss(x_hat, x.reshape(x.shape[0], -1), reduction='none').mean(dim=-1)
        y_hat = torch.where(all_mse > self.threshold, torch.zeros_like(y), torch.ones_like(y))
        acc = accuracy(y_hat, y, task='binary')
        # TODO: add auroc
        return loss, acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


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
    

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class ConvAE(pl.LightningModule):
    def __init__(self,
                 input_channels: int,
                 height : int,
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
                 pool_kernel : int = None,
                 lr: float = 1e-3,
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
        self.height = height
        self.lr = lr
        
        self.threshold = 0

        # check that the scaling factor is valid
        if not (0 < scaling_factor < 1):
            raise ValueError(f'Invalid scaling factor: {scaling_factor}')
        
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
                    conv_type='downsample',
                    pool_kernel=self.pool_kernel
                )
            )
     
        blocks.append(nn.Flatten())
        num_channels = self.encoder_width
        dims = int(height*(1/self.pool_kernel)**(self.n_layers_encoder))
        blocks.append(nn.Linear(num_channels * dims * dims, self.latent_dim))
        self.encoder = nn.Sequential(*blocks)

        # Decoder
        decoder_dim = round(self.height/2**self.n_layers_decoder)
        blocks_dec = [nn.Linear(self.latent_dim, num_channels * decoder_dim * decoder_dim), 
                  Reshape(-1, num_channels, decoder_dim, decoder_dim)] 
        for i in range(self.n_layers_decoder):
            H_in = round(height*(1/self.pool_kernel)**(self.n_layers_decoder - i))
            H_out = round(height*(1/self.pool_kernel)**(self.n_layers_decoder - (i+1)))
            stride = 2 # 2 for upsampling with conv_transpose
            padding = self.padding
            dilation = 1
            kernel_size = self.kernel_size
            output_padding = H_out - ((H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1)

            in_channels = num_channels if i == 0 else int(decoder_width * scaling_factor ** (i - 1))
            out_channels = input_channels if i == self.n_layers_decoder - 1 else int(decoder_width * scaling_factor ** (i))

            activation = nn.Sigmoid if i == self.n_layers_decoder - 1 else nn.ReLU
            norm = None if i == self.n_layers_decoder - 1 else self.norm

            blocks_dec.append(
                self.block(
                    in_channels,
                    out_channels,
                    norm=norm, dropout=dropout, output_padding=output_padding,
                    conv_type='upsample',
                    stride=stride,
                    activation=activation
                )
            )

        self.decoder = nn.Sequential(*blocks_dec)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        self.threshold = F.mse_loss(x_hat, x, reduction = 'mean')
        self.log('threshold', self.threshold)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        # # get indexes of class we train on
        # normal_class_idx = torch.where(y == self.normal_class)[0]
 
        # x_hat = self(x)
        # # get loss of model only for the class that we trained on
        # loss = F.mse_loss(
        #     x_hat[normal_class_idx],
        #     x[normal_class_idx]
        # )
        # # get reconstruction error for every example
        # all_mse = F.mse_loss(x_hat.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none').mean(dim=-1)

        # # get classification based on threshold
        # y_hat = torch.where(all_mse > self.threshold, torch.ones_like(y), torch.zeros_like(y))

        # # get anomaly accuracy
        # acc = accuracy(
        #     y_hat,
        #     torch.where(y == self.normal_class, torch.zeros_like(y), torch.ones_like(y)),
        #     task='binary'
        # )
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x.reshape(x.shape[0], -1))

        all_mse = F.mse_loss(x_hat, x.reshape(x.shape[0], -1), reduction='none').mean(dim=-1)
        y_hat = torch.where(all_mse > self.threshold, torch.zeros_like(y), torch.ones_like(y))
        acc = accuracy(y_hat, y, task='binary')

        # TODO: add auroc
        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, input):
        z = self.encoder(input)
        x_hat = self.decoder(z)
        return x_hat

class ConvVAE(ConvAE):
    def __init__(self, 
                   *args,
                    **kwargs):
        super().__init__(*args, **kwargs)
    
        # Encoder
        blocks = []
        for i in range(self.n_layers_encoder):
            in_channels = self.input_dim if i == 0 else int(self.encoder_width * self.scaling_factor ** (self.n_layers_encoder - i))
            out_channels = int(self.encoder_width * self.scaling_factor ** (self.n_layers_encoder - (i + 1)))
            blocks.append(
                self.block(
                    in_channels,
                    out_channels,
                    norm=self.norm, dropout=self.dropout,
                    conv_type='downsample',
                    pool_kernel=self.pool_kernel
                )
            )
     
        blocks.append(nn.Flatten()) # or just make encoder blocks = self.blocks, and pop out linear layer
        self.encoder = nn.Sequential(*blocks)

        num_channels = self.encoder_width
        dims = int(self.height*(1/self.pool_kernel)**(self.n_layers_encoder))

        self.mu = nn.Linear(dims * dims * num_channels, self.latent_dim)
        self.log_variance = nn.Linear(dims * dims * num_channels, self.latent_dim)
        self.VAE_loss = VAELoss()

    def sample_noise(self):
        return torch.randn(self.latent_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        log_variance = self.log_variance(x)
        epsilon = self.sample_noise()
        z = mu + torch.exp(0.5*log_variance) * epsilon
        x = self.decoder(z)
        return x, mu, log_variance
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = self.VAE_loss(x_hat, x)
        self.log('train_loss', loss)
        self.threshold = self.VAE_Loss.reconstruction_loss(x_hat, x)
        self.log('threshold', self.threshold)
        return loss

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch

        x_hat = self(x)
        loss = (x_hat, x.reshape(x.shape[0], -1))

        all_mse = self.VAE_Loss.reconstruction_loss(x_hat, x.reshape(x.shape[0], -1), reduction='none').mean(dim=-1)
        y_hat = torch.where(all_mse > self.threshold, torch.zeros_like(y), torch.ones_like(y))
        acc = accuracy(y_hat, y, task='binary')

        # TODO: add auroc
        return loss, acc
    

class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.alpha = 1100

    def kl_loss(self, mu, log_variance):
        loss = -0.5 * torch.sum(1 + log_variance - torch.square(mu) - torch.exp(log_variance), dim=1)
        return loss

    def reconstruction_loss(self, y_true, y_pred):
        loss = torch.mean(torch.square(y_true - y_pred), dim=(1, 2, 3))
        return loss

    def forward(self, predictions, targets, mu, log_variance):
        kld = self.kl_loss(mu, log_variance)
        reconstruction = self.reconstruction_loss(targets, predictions)
        return torch.sum(kld + self.alpha*reconstruction)


class ConvBlock(nn.Module):
    def __init__(self, 
                 input_dim : int,
                 output_dim : int,
                 pool_kernel : int = 2,
                 kernel_size : int = 3,
                 stride : int = 1,
                 padding : int = 1,
                 activation : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU, # TODO: Change to leakyrelu
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
                          stride=1,
                          padding=padding),
                activation(),
                nn.MaxPool2d(pool_kernel) 
            )

        else:
            upsample_stride = pool_kernel
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(input_dim,
                                output_dim,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=upsample_stride, 
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
        'height' : [28],
        "n_layers_encoder" : [2],
        'encoder_width' : [4], # max num of channels
        'n_layers_decoder' : [3],
        'decoder_width' : [8], # max num of channels
        'latent_dim' : [32], # FC 
        'scaling_factor' : [1/2],
        'norm' : ['batch'],
        'dropout' : [0.3],
        'padding' : [1],
        'kernel_size' : [3],
        'pool_kernel' : [2],
    }

    # for i in range(12):
    #     args_ = {k: v[i] for k, v in MLP_args.items()}
    #     print(args_)
    #     model = DenseAE(**args_)
    #     input = torch.randn(2, args_['input_dim'])
    #     output = model(input)
    #     assert output.shape == input.shape, "Autoencoder output shape does not match input shape"

    args_ = {k: v[0] for k, v in CNN_args.items()}
    print(args_)
    model = ConvVAE(**args_)
    input = torch.randn(2, 1, 28, 28)
    output = model(input)
    assert output[0].shape == input.shape, "Autoencoder output shape does not match input shape"
    print(model)
        