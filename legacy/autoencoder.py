import torch
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from models import Autoencoder
from tqdm import tqdm

cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")


def main():
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, 
                                transform=transforms.ToTensor())

    batch_size = 64
    dataloader = data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

    model = Autoencoder().to(device)
    loss = nn.MSELoss()
    learning_rate = 1e-3
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 10

    for epoch in range(epochs):

        for i, train_data in enumerate(tqdm(dataloader)):
            x, _ = train_data

            x = x.to(device)
            xtilde = model(x)
            recon_error = loss(xtilde, x)

            optim.zero_grad()
            recon_error.backward()
            optim.step()

        if (epoch + 1) % 2 == 0:
            plt.subplot(1, 2, 1)
            plt.imshow(xtilde[0].squeeze().detach().cpu().numpy())

            plt.subplot(1, 2, 2)
            plt.imshow(x[0].squeeze().detach().cpu().numpy())

            plt.show()
            
    torch.save(model, 'models/small_AE.pt')

if __name__ == '__main__':
    main()