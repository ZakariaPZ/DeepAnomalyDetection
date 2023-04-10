from torch import nn
import torch

cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")



class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.latent_dim = 8
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # 14 x 14 x 16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 7 x 7 x 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 4 x 4 x 64
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(4 * 4 * 64, self.latent_dim)
        )
        
    def forward(self, x):
        
        return self.encoder(x)
    
class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.latent_dim = 8
        
        self.linear = nn.Linear(self.latent_dim, 4 * 4 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1), # 7 x 7 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 14 x 14 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), # 28 x 28 x 1
            nn.Sigmoid()
        )
        
    def forward(self, x):

        x = self.linear(x)
        x = x.view(-1, 64, 4, 4)
        return self.decoder(x)
    
class Autoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x) 
        x = self.decoder(x)
        return x
    
   

class VariationalEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.latent_dim = 8

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # 14 x 14 x 16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 7 x 7 x 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 4 x 4 x 64
            nn.ReLU(),
            nn.Flatten()
        )

        self.mu = nn.Linear(4 * 4 * 64, self.latent_dim)
        self.log_variance = nn.Linear(4 * 4 * 64, self.latent_dim)

    def sample_noise(self):
        return torch.randn(self.latent_dim)

    def forward(self, x):
        
        x = self.encoder(x)
        mu = self.mu(x)
        log_variance = self.log_variance(x)
        epsilon = self.sample_noise().to(device)
        z = mu + torch.exp(0.5*log_variance) * epsilon

        return z, mu, log_variance
    
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
    
class VariationalAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = VariationalEncoder()
        self.decoder = Decoder()

    def forward(self, x):
        x, mu, log_variance = self.encoder(x) 
        x = self.decoder(x)
        return x, mu, log_variance
    
