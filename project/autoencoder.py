import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels, sn=False):
        super().__init__()

        modules = []

        # DONE:
        #  Implement a CNN. Save the layers in the modules list.
        #  The input shape is an image batch: (N, in_channels, H_in, W_in).
        #  The output shape should be (N, out_channels, H_out, W_out).
        #  You can assume H_in, W_in >= 64.
        #  Architecture is up to you, but it's recommended to use at
        #  least 3 conv layers. You can use any Conv layer parameters,
        #  use pooling or only strides, use any activation functions,
        #  use BN or Dropout, etc.
        # ====== YOUR CODE: ======
        # implementing an architecture inspired by the paper referred to in the note book
        channels = [32, 64, 128, 256]
        
        for curr_channel in channels:
            if sn:
                modules.append(nn.utils.spectral_norm(nn.Conv2d(in_channels, curr_channel, 5, stride=2, padding=2)))
            else:
                modules.append(nn.Conv2d(in_channels, curr_channel, 5, stride=2, padding=2))
            modules.append(nn.BatchNorm2d(curr_channel))
            modules.append(nn.ReLU())
            in_channels = curr_channel

        # layer 5
        modules.append(nn.Conv2d(curr_channel, out_channels, 5, padding=2))
        modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.ReLU())
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # DONE:
        #  Implement the "mirror" CNN of the encoder.
        #  For example, instead of Conv layers use transposed convolutions,
        #  instead of pooling do unpooling (if relevant) and so on.
        #  The architecture does not have to exactly mirror the encoder
        #  (although you can), however the important thing is that the
        #  output should be a batch of images, with same dimensions as the
        #  inputs to the Encoder were.
        # ====== YOUR CODE: ======
        # implementing an architecture inspired by the paper referred to in the note book
        # layer 1 (decodes layer 4)
        channels = [256, 128, 64, 32]
        
        for curr_channel in channels:
            modules.append(nn.ConvTranspose2d(in_channels, curr_channel, 5, stride=2, padding=2, output_padding=1))
            modules.append(nn.BatchNorm2d(curr_channel))
            modules.append(nn.ReLU())
            in_channels = curr_channel
        
        modules.append(nn.ConvTranspose2d(curr_channel, out_channels, 5, stride=1, padding=2))
        modules.append(nn.BatchNorm2d(out_channels))
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        # DONE: Add more layers as needed for encode() and decode().
        # ====== YOUR CODE: ======
        self.mu = nn.Linear(n_features, self.z_dim)
        self.log_sigma2 = nn.Linear(n_features, self.z_dim)
        self.reconstruct = nn.Linear(self.z_dim, n_features)
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        # DONE:
        #  Sample a latent vector z given an input x from the posterior q(Z|x).
        #  1. Use the features extracted from the input to obtain mu and
        #     log_sigma2 (mean and log variance) of q(Z|x).
        #  2. Apply the reparametrization trick to obtain z.
        # ====== YOUR CODE: ======
        N = x.shape[0]

        # obtaining mu and sigma
        h = self.features_encoder(x)
        h = h.reshape((N, -1))
        mu = self.mu(h)
        log_sigma2 = self.log_sigma2(h)

        # obtaining z
        mean = torch.zeros_like(mu)
        std = torch.ones_like(mu)
        u = torch.normal(mean, std)
        sigma = torch.exp(log_sigma2)
        z = mu + sigma * u
        # ========================

        return z, mu, log_sigma2

    def decode(self, z):
        # DONE:
        #  Convert a latent vector back into a reconstructed input.
        #  1. Convert latent z to features h with a linear layer.
        #  2. Apply features decoder.
        # ====== YOUR CODE: ======
        z = self.reconstruct(z)
        h = z.reshape((-1,) + self.features_shape)
        x_rec = self.features_decoder(h)
        # ========================

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # DONE:
            #  Sample from the model. Generate n latent space samples and
            #  return their reconstructions.
            #  Notes:
            #  - Remember that this means using the model for INFERENCE.
            #  - We'll ignore the sigma2 parameter here:
            #    Instead of sampling from N(psi(z), sigma2 I), we'll just take
            #    the mean, i.e. psi(z).
            # ====== YOUR CODE: ======
            # z ~ N(0,1)
            for i in range(n):
                mean = torch.zeros(self.z_dim, device=device)
                std = torch.ones_like(mean)
                z = torch.normal(mean, std)
                samples.append(self.decode(z).squeeze().cpu())
            # ========================

        # Detach and move to CPU for display purposes
        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    # DONE:
    #  Implement the VAE pointwise loss calculation.
    #  Remember:
    #  1. The covariance matrix of the posterior is diagonal.
    #  2. You need to average over the batch dimension.
    # ====== YOUR CODE: ======
    N = x.shape[0]
    z_dim_sigma = z_log_sigma2.shape[-1]
    z_dim_mu = z_mu.shape[-1]

    # data loss
    l2norm = nn.MSELoss()
    data_loss = l2norm(x, xr)
    data_loss = (1 / x_sigma2) * data_loss

    # KL divergence loss
    sum_batches = torch.zeros((N, z_dim_sigma, z_dim_sigma))
    z_sigma = torch.exp(z_log_sigma2)
    # populating the matrix diagonal
    sum_batches.as_strided(z_log_sigma2.size(), [sum_batches.stride(0), z_dim_sigma + 1]).copy_(z_sigma)

    kldiv_list = []

    for i in range(N):
        kldiv_curr = torch.trace(sum_batches[i]) + torch.norm(z_mu[i]) ** 2 - z_dim_mu - torch.log(
            torch.det(sum_batches[i]))
        kldiv_list.append(kldiv_curr)

    kldiv_loss = torch.sum(torch.stack(kldiv_list)) / N

    # VAE loss
    loss = data_loss + kldiv_loss
    # ========================

    return loss, data_loss, kldiv_loss
