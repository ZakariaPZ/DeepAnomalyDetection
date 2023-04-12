from typing import *

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from ae import DenseAE
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy, ConfusionMatrix
from torchmetrics.functional import confusion_matrix
from torchvision import transforms
from torchvision.datasets import MNIST

BATCH_SIZE = 256
TRAIN_DIGIT = 8
COMPARE_DIGIT = 7
VAL_SIZE = 0.5

if __name__ == '__main__':
    pl.seed_everything(1)

    model = DenseAE(
        input_dim = 28*28,
        latent_dim = 32,
        n_layers_encoder = 2,
        n_layers_decoder = 2,
        encoder_width = 784,
        decoder_width = 784,
        scaling_factor = 1/2,
        norm = 'batch',
        dropout = 0.3,
        normal_class=TRAIN_DIGIT
    )

    train = MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    # only use the digit we want
    train_idx = torch.where(train.targets == TRAIN_DIGIT)[0]
    train.targets = train.targets[train_idx]
    train.data = train.data[train_idx]
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
    

    test = MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    # split the dataset into train and val
    n_val = int(len(test) * VAL_SIZE)
    n_test = len(test) - n_val
    test, val = random_split(test, [n_test, n_val])
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)

    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        max_epochs=10,
        check_val_every_n_epoch=1,
        enable_checkpointing=False,
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)
    # visulize a batch item and its reconstruction
    batch = next(iter(test_loader))
    model.eval()

    idx = torch.where(batch[1] == TRAIN_DIGIT)[0][0]

    plt.subplot(121)
    plt.imshow(batch[0][idx].squeeze(), cmap='gray')
    plt.title(f'Original: {batch[1][idx]}')

    plt.subplot(122)
    plt.imshow(model(batch[0][idx]).squeeze().reshape(batch[0][0].shape[1], -1).detach().numpy(), cmap='gray')
    plt.title(f'Reconstruction: {batch[1][idx]}')
    plt.savefig(f'reconstruction{TRAIN_DIGIT}.png', bbox_inches='tight')
    plt.show()

    idx = torch.where(batch[1] == COMPARE_DIGIT)[0][0]

    plt.subplot(121)
    plt.imshow(batch[0][idx].squeeze(), cmap='gray')
    plt.title(f'Original: {batch[1][idx]}')

    plt.subplot(122)
    plt.imshow(model(batch[0][idx]).squeeze().reshape(batch[0][0].shape[1], -1).detach().numpy(), cmap='gray')
    plt.title(f'Reconstruction: {batch[1][idx]}')
    plt.savefig(f'reconstruction{COMPARE_DIGIT}.png', bbox_inches='tight')
    plt.show()

    # for each batch in the test set, calculate the confusion matrix
    for idx, batch in enumerate(test_loader):
        x, y = batch
        # get indexes of class we train on
        normal_class_idx = torch.where(y == model.normal_class)[0]
                
        x_hat = model(x)

        # get reconstruction error for every example
        all_mse = F.mse_loss(x_hat, x.reshape(x.shape[0], -1), reduction='none').mean(dim=-1).to(x.device)

        # get classification based on threshold
        y_hat = torch.where(all_mse > model.threshold.to(x.device), torch.ones_like(y, device=x.device), torch.zeros_like(y, device=x.device))
        
        # get anomaly accuracy
        conf = confusion_matrix(
            y_hat,
            y,
            task='multiclass',
            num_classes=10,
        )

        if idx == 0:
            confusion = conf
        else:
            confusion += conf
        
    print(confusion)