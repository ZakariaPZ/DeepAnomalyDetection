from torch import nn


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