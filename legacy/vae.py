import torch
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from models import Autoencoder, VAELoss, VariationalAutoencoder
from tqdm import tqdm


cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")


def main():
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, 
                                transform=transforms.ToTensor())

    batch_size = 64
    dataloader = data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

    model = VariationalAutoencoder().to(device)
    loss = VAELoss()

    learning_rate = 1e-4
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 30

    for epoch in range(epochs):

        for i, train_data in enumerate(tqdm(dataloader)):
            x, _ = train_data

            x = x.to(device)
            xtilde, mu, log_variance = model(x)
            error = loss(xtilde, x, mu, log_variance)

            optim.zero_grad()
            error.backward()
            optim.step()

    torch.save(model, 'models/small_VAE.pt')

if __name__ == '__main__':
    main()