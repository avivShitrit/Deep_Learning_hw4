import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from .autoencoder import EncoderCNN, DecoderCNN

class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        C = self.in_size[0]
        out_channels = 1024
        self.encoder = EncoderCNN(C, out_channels)
        self.linear_layer = nn.Linear(out_channels, 1)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        batch_size = x.shape[0]
        
        encoded = self.encoder(x)
        kernel_size = encoded.shape[2:]
        encoded = F.max_pool2d(encoded, kernel_size=kernel_size).view(-1, 1024)
        
        y = self.linear_layer(encoded).view(batch_size, 1)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        self.hidden_dim = 256
        prep_layer_mid_size = self.hidden_dim * featuremap_size ** 2
        prep_layer_out_size = self.hidden_dim * 4 ** 2
        
        # Transform the input size to the same size (256x5x5)
        self.preperation_layer = nn.Sequential(
            nn.Linear(z_dim, prep_layer_mid_size),
            nn.Linear(prep_layer_mid_size, prep_layer_out_size)
        )
        
        self.decoder = DecoderCNN(self.hidden_dim, out_channels)
        
        self.featuremap_dim = (self.hidden_dim, 4, 4)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        torch.set_grad_enabled(with_grad)
        
        mean = torch.zeros((n, self.z_dim), device=device, requires_grad=with_grad)
        std = torch.ones((n, self.z_dim), device=device, requires_grad=with_grad)
        samples = self.forward(torch.normal(mean,std))
        
        torch.set_grad_enabled(True)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        batch_size = z.shape[0]
        in_decoder_size = (batch_size, ) + self.featuremap_dim
        x = self.preperation_layer(z).reshape(in_decoder_size)
        x = self.decoder(x)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    batch_size = y_data.shape[0]
    BCWloss = nn.BCEWithLogitsLoss()
    label_noise /= 2
    
    real_uniform = torch.distributions.uniform.Uniform(data_label - label_noise, data_label + label_noise)
    fake_uniform = torch.distributions.uniform.Uniform(1 - data_label - label_noise, 1 - data_label + label_noise)
    
    real_labels = real_uniform.sample((batch_size, )).to(y_data.device)
    fake_labels = fake_uniform.sample((batch_size, )).to(y_data.device)
    
    loss_data = BCWloss(torch.squeeze(y_data), real_labels)
    loss_generated = BCWloss(torch.squeeze(y_generated), fake_labels)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    fake_labels = torch.full_like(y_generated, data_label, device=y_generated.device, requires_grad=True)
    BCWloss = nn.BCEWithLogitsLoss()
    
    loss = BCWloss(y_generated, fake_labels)
    # ========================
    return loss


def wgan_discriminator_loss_fn(y_data, y_generated):
    loss = -torch.mean(y_data) + torch.mean(y_generated)
    return loss

def wgan_generator_loss_fn(y_generated):
    loss = -torch.mean(y_generated)
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: DataLoader,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()
    batch_size = x_data.shape[0]
    generated_data = gen_model.sample(batch_size, False)
    real = dsc_model(x_data)
    fake = dsc_model(generated_data)
    dsc_loss = dsc_loss_fn(real, fake)
    dsc_loss.backward()
    dsc_optimizer.step()
    
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    for _ in range(2):
        gen_optimizer.zero_grad()

        x_z = gen_model.sample(batch_size, True)
        score = dsc_model(x_z)
        gen_loss = gen_loss_fn(score)
        gen_loss.backward()
        gen_optimizer.step()
    
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    epoch = len(dsc_losses)
    checkpoint_file = f"{checkpoint_file}_{epoch}.pt"
    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    avg_dsc_loss = dsc_losses[-1]
    avg_gen_loss = gen_losses[-1]
    
    if(epoch>20):
        torch.save(gen_model, checkpoint_file)
        print("saved checkpoint")
        saved = True

    # ========================

    return saved

class Discriminator_SN(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        C = self.in_size[0]
        out_channels = 1024
        self.encoder = EncoderCNN(C, out_channels, sn=True)
        self.linear_layer = nn.utils.spectral_norm(nn.Linear(out_channels, 1))
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        batch_size = x.shape[0]
        
        encoded = self.encoder(x)
        kernel_size = encoded.shape[2:]
        encoded = F.max_pool2d(encoded, kernel_size=kernel_size).view(-1, 1024)
        
        y = self.linear_layer(encoded).view(batch_size, 1)
        # ========================
        return y