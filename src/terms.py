from lightning_toolbox.objective_function import ObjectiveTerm
import torch


class ReconstructionLossTerm(ObjectiveTerm):
    def __call__(self, training_module, batch, **kwargs):
        inputs, _ = batch
        latent = training_module.model.encoder(inputs)
        self.remember(latent=latent)  # save for later use (e.g. in a callback or in another term)
        # the model forward pass takes care of the reparameterization if needed
        reconstruction = training_module(latent=latent)
        self.remember(reconstruction=reconstruction)  # save for later use (e.g. in a callback)
        return torch.nn.functional.mse_loss(reconstruction, inputs, reduction="none").mean(dim=(1, 2, 3))


class KLDLossTerm(ObjectiveTerm):
    def __call__(self, training_module, batch, **kwargs):
        latent = self.objective.latch.get("latent", None)
        if latent is None:
            latent = training_module.model.encoder(batch[0])
        # split latent into mu and log_variance
        mu, log_variance = torch.chunk(latent, 2, dim=1)
        self.remember(mu=mu, log_variance=log_variance)  # save for later use (e.g. in a callback)
        # compute KL divergence
        return -0.5 * torch.sum(1 + log_variance - torch.square(mu) - torch.exp(log_variance), dim=1)
