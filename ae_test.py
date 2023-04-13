import argparse
from typing import *

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from ae import ConvAE, DenseAE
from pytorch_lightning.loggers import TensorBoardLogger
from src.callbacks import NoveltyAUROCCallback
from src.data import NoveltyDetectionDatamodule
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import confusion_matrix
from torchvision import transforms
from torchvision.datasets import MNIST

if __name__ == '__main__':
    pl.seed_everything(1)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        type=str,
        default='dense',
        choices=['dense', 'conv'],
        help='type of autoencoder to use',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='MNIST',
        choices=['MNIST', 'CIFAR10'],
        help='dataset to use',
    )
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=32,
        help='latent dimension of the autoencoder',
    )
    parser.add_argument(
        '--n_layers_encoder',
        type=int,
        default=2,
        help='number of layers in the encoder (not including the first and last linear map layers)',
    )
    parser.add_argument(
        '--n_layers_decoder',
        type=int,
        default=2,
        help='number of layers in the decoder (not including the first and last linear map layers)',
    )
    parser.add_argument(
        '--encoder_width',
        type=int,
        default=784,
        help='width of the first encoder layer after the input layer (in neurons for dense or num channels for CNN)',
    )
    parser.add_argument(
        '--decoder_width',
        type=int,
        default=784,
        help='width of the last decoder layer before the output layer',
    )
    parser.add_argument(
        '--scaling_factor',
        type=float,
        default=1/2,
        help='scaling factor for the width of the encoder and decoder layers',
    )
    parser.add_argument(
        '--norm',
        type=str,
        default='batch',
        help='normalization to use in the encoder and decoder',
    )
    parser.add_argument(
        '--dropout',
        default=0.3,
        help='dropout to use in the encoder and decoder',
    )
    parser.add_argument(
        '--normal_class',
        type=int,
        default=1,
        help='class to use as the normal class',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256 if torch.cuda.is_available() else 64,
        help='batch size to use for training',
    )
    parser.add_argument(
        '--val_size',
        type=float,
        default=0.2,
        help='proportion of the train set to use as the validation set',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='learning rate to use for training',
    )
    parser.add_argument(
        '--eval_class',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
    )
    parser.add_argument(
        '--checkpoint',
        action='store_true',
    )
    parser.add_argument(
        '--version',
        type=int,
        help='version of the model to save for TensorBoard',
    )
    parser.add_argument(
        '--kernel_size',
        type=int,
        default=3,
    )

    args = parser.parse_args()

    # check norm and dropout args
    if args.norm not in ['batch', 'layer', 'None']:
        raise ValueError(f'Invalid norm: {args.norm}')
    else:
        args.norm = None if args.norm == 'None' else args.norm
    
    if args.dropout == 'None':
        args.dropout = None
    else:
        args.dropout = float(args.dropout)
        if args.dropout < 0 or args.dropout > 1:
            raise ValueError(f'Invalid dropout: {args.dropout}')

  
    if args.type == 'dense':

        if args.dataset == 'MNIST':
            input_dim = 28 * 28
        else:
            input_dim = 32 * 32 * 3

        model = DenseAE(
            input_dim = input_dim,
            latent_dim = args.latent_dim,
            n_layers_encoder = args.n_layers_encoder,
            n_layers_decoder = args.n_layers_decoder,
            encoder_width = args.encoder_width,
            decoder_width = args.decoder_width,
            scaling_factor = args.scaling_factor,
            norm = args.norm,
            dropout = args.dropout,
            normal_class=args.normal_class,
            lr=args.lr,
        )
    else:
        if args.dataset == 'MNIST':
            input_channels = 1
            height = 28
        else:
            input_channels = 3
            height = 32

        model = ConvAE(
            normal_class=args.normal_class,
            input_channels= input_channels,
            height = height,
            latent_dim = args.latent_dim,
            n_layers_encoder = args.n_layers_encoder,
            n_layers_decoder = args.n_layers_decoder,
            encoder_width = args.encoder_width,
            decoder_width = args.decoder_width,
            scaling_factor = args.scaling_factor,
            norm = args.norm,
            dropout = args.dropout,
            padding = 1,
            kernel_size=args.kernel_size,
            pool_kernel=2,
            lr=args.lr,
        )

    # TODO: add handling for CIFAR10
    dm = NoveltyDetectionDatamodule(
        dataset="torchvision.datasets.MNIST",
        dataset_args=dict(
            root="data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),  # TODO: add transforms to the base datamodule to be configurable from cli
        ),
        test_dataset="torchvision.datasets.MNIST",
        test_dataset_args=dict(
            root="data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),  # TODO: add transforms to the base datamodule to be configurable from cli
        ),
        val_size = args.val_size,
        normal_targets=[args.normal_class],
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )

    dm.setup('fit')
    dm.setup('test')
     

    ##### OLD DATA LOADING #####
    # train = MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    # # only use the digit we want
    # train_idx = torch.where(train.targets == args.normal_class)[0]
    # train.targets = train.targets[train_idx]
    # train.data = train.data[train_idx]
    # train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=2, shuffle=True)
    # 
    # test = MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    # # split the dataset into train and val
    # n_val = int(len(test) * args.val_size)
    # n_test = len(test) - n_val
    # test, val = random_split(test, [n_test, n_val])
    # test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=2, shuffle=False)
    # val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=2, shuffle=False)
    ##### END OLD DATA LOADING #####

    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        enable_checkpointing=args.checkpoint,
        log_every_n_steps=10,
        enable_progress_bar=False,
        logger=TensorBoardLogger('lightning_logs', name=args.type, version=args.version),
        # callbacks=[NoveltyAUROCCallback()],
    )

    trainer.fit(model, dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    trainer.test(model, dataloaders=dm.test_dataloader())

    # visulize a batch item and its reconstruction
    # batch = next(iter(test_loader))
    # model.eval()

    # idx = torch.where(batch[1] == args.normal_class)[0][0]

    # plt.subplot(121)
    # plt.imshow(batch[0][idx].squeeze(), cmap='gray')
    # plt.title(f'Original: {batch[1][idx]}')

    # plt.subplot(122)
    # plt.imshow(model(batch[0][idx]).squeeze().reshape(batch[0][0].shape[1], -1).detach().numpy(), cmap='gray')
    # plt.title(f'Reconstruction: {batch[1][idx]}')
    # plt.savefig(f'reconstruction{args.normal_class}.png', bbox_inches='tight')
    # plt.show()

    # idx = torch.where(batch[1] == args.eval_class)[0][0]

    # plt.subplot(121)
    # plt.imshow(batch[0][idx].squeeze(), cmap='gray')
    # plt.title(f'Original: {batch[1][idx]}')

    # plt.subplot(122)
    # plt.imshow(model(batch[0][idx]).squeeze().reshape(batch[0][0].shape[1], -1).detach().numpy(), cmap='gray')
    # plt.title(f'Reconstruction: {batch[1][idx]}')
    # plt.savefig(f'reconstruction{args.eval_class}.png', bbox_inches='tight')
    # plt.show()

    # # for each batch in the test set, calculate the confusion matrix
    # for idx, batch in enumerate(test_loader):
    #     x, y = batch
    #     # get indexes of class we train on
    #     normal_class_idx = torch.where(y == model.normal_class)[0]
                
    #     x_hat = model(x)

    #     # get reconstruction error for every example
    #     all_mse = F.mse_loss(x_hat, x.reshape(x.shape[0], -1), reduction='none').mean(dim=-1).to(x.device)

    #     # get classification based on threshold
    #     y_hat = torch.where(all_mse > model.threshold.to(x.device), torch.ones_like(y, device=x.device), torch.zeros_like(y, device=x.device))
        
    #     # get anomaly accuracy
    #     conf = confusion_matrix(
    #         y_hat,
    #         y,
    #         task='multiclass',
    #         num_classes=10,
    #     )

    #     if idx == 0:
    #         confusion = conf
    #     else:
    #         confusion += conf
        
    # print(confusion)