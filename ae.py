from typing import *

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy, confusion_matrix


class DenseAE(pl.LightningModule):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 n_layers_encoder: int,
                 n_layers_decoder: int,
                 encoder_width: int,
                 decoder_width: int,
                 normal_class: int,
                 scaling_factor: int = 1/2.,
                 norm: Union[str, None] = None,
                 dropout: Union[float, None] = None,
                 type: str = 'dense',
                ):
        super().__init__()

        self.normal_class = normal_class
        self.threshold = 0

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
        # loss = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, prog_bar=True)
        # self.log('val_loss', loss, prog_bar=True)
        return metrics
        # return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        # loss = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, prog_bar=True)
        # self.log('test_loss', loss, prog_bar=True)
        return metrics
        # return loss

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        # get indexes of class we train on
        normal_class_idx = torch.where(y == self.normal_class)[0]
                
        x_hat = self(x)
        # get loss of model only for the class that we trained on
        loss = F.mse_loss(
            x_hat[normal_class_idx],
            x[normal_class_idx]\
                .reshape(normal_class_idx.shape[0], -1)
        )
        # get reconstruction error for every example
        all_mse = F.mse_loss(x_hat, x.reshape(x.shape[0], -1), reduction='none').mean(dim=-1)

        # get classification based on threshold
        y_hat = torch.where(all_mse > self.threshold, torch.ones_like(y), torch.zeros_like(y))
        
        # get anomaly accuracy
        acc = accuracy(
            y_hat,
            torch.where(y == self.normal_class, torch.zeros_like(y), torch.ones_like(y)),
            task='binary'
        )

        return loss, acc
        # return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    

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
        model = DenseAE(**args_, normal_class=0)
        input = torch.randn(2, args_['input_dim'])
        output = model(input)
        assert output.shape == input.shape, "Autoencoder output shape does not match input shape"
        print(model)
        exit()
        