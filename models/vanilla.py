import torch
import torch.nn.functional as F
from torch import nn


from .base import BaseVAE
from .types import *


class VanillaVAE(BaseVAE):

    def __init__(self, in_channels: int, latent_dim: int,
                 hidden_dims: List = None, **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]  # 256, 512]

        # Build Encoder module
        for dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=dim, kernel_size=3,
                              stride=2, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = dim
        self.encoder = nn.Sequential(*modules)

        # Mappings to latent dimension for mean and variance
        num_flat_features = 1024 # hidden_dims[-1] * 4
        self.fc_mu = nn.Linear(num_flat_features, latent_dim)
        self.fc_var = nn.Linear(num_flat_features, latent_dim)

        # Build Decoder module
        modules = []
        self.decoder_input = nn.Linear(latent_dim, num_flat_features)
        hidden_dims.reverse()

        for input_ch, output_ch in zip(hidden_dims, hidden_dims[1:]):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(input_ch, output_ch, kernel_size=3,
                                       stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(output_ch),
                    nn.LeakyReLU()
                )
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3,
                      padding=1),
            nn.Tanh()
        )

    def encode(self, inputs: Tensor) -> List[Tensor]:
        outputs = self.encoder(inputs)
        # flatten everything aside from batch dimension
        outputs = torch.flatten(outputs, start_dim=1)
        mu = self.fc_mu(outputs)
        log_var = self.fc_var(outputs)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        outputs = self.decoder_input(z).view(-1, 256, 2, 2)
        outputs = self.decoder(outputs)
        outputs = self.final_layer(outputs)
        return outputs

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick to samples from `Normal(mu, var)` from
        `Normal(0, 1)`.
        :param mu: (Tensor) mean of the latent Gaussian [B x D]
        :param log_var: (Tensor) standard deviation of latent Gaussian [B x D]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, inputs: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        return [recons, inputs, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function
        """
        recons = args[0]
        inputs = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = F.mse_loss(recons, inputs)
        kld_weight = kwargs['M_N']
        kld_loss = torch.mean(-0.5*torch.sum(1 + log_var - mu**2 -
            log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'MSE': recons_loss, 'KLD': -kld_loss}

    def sample(self, num_samples:int, current_device:int, **kwargs) -> Tensor:
        """
        Samples tensor from latent space, and returns reconstructed output of the tensor
        in the image space.

        :param num_samples:
        :param current_device:
        :param kwargs:
        :return:
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Returns reconstruction image(s) for a given batch of images.

        :param x: (Tensor) [B x C x H x W]
        :param kwargs:
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]

