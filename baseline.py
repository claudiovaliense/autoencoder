import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from torch.nn import functional as F
from torchvision import transforms

import pytorch_lightning as pl

from torchvision.datasets import MNIST

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./', batch_size=64, num_workers=4):

        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # We hardcode dataset specific stuff here.
        self.num_classes = 10
        self.dims = (1, 28, 28)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

class SimpleAutoEncoder(pl.LightningModule):
  def __init__(self, input_shape, representation_size):
    super().__init__()

    self.save_hyperparameters() # Saves the hyperparams -- input_shape, representation_size

    self.input_shape = input_shape
    self.representation_size = representation_size

    # Calculate the flattened size
    flattened_size = 1
    for x in self.input_shape:
      flattened_size *= x

    self.flattened_size = flattened_size

    # Initialise the Dense Layers
    self.input_to_representation = nn.Linear(self.flattened_size, self.representation_size)
    self.representation_to_output = nn.Linear(self.representation_size, self.flattened_size)


  def forward(self, image_batch):
    ## ENCODING
    # image_batch: [batch_size, ...] -- Other dimensions are the input_shape
    flattened = image_batch.view(-1, self.flattened_size)
    # flattened: [batch_size, flattened_size]
    representation = F.relu(self.input_to_representation(flattened))
    # representation: [batch_size, representation_size]

    ## DECODING
    flat_reconstructed = F.relu(self.representation_to_output(representation))
    # flat_reconstructed: [batch_size, flattened_size]
    reconstructed = flat_reconstructed.view(-1, *self.input_shape)
    # reconstructed is same shape as image_batch

    return reconstructed


  def training_step(self, batch, batch_idx):
    batch_images = batch[0]
    # Get the reconstructed images
    reconstructed_images = self.forward(batch_images)
    # Calculate loss
    batch_loss = F.mse_loss(reconstructed_images, batch_images)

    # store the result
    result = pl.TrainResult(minimize=batch_loss)
    result.batch_loss = batch_loss
    result.log('train_loss', batch_loss, prog_bar=True)

    return result


  def validation_step(self, batch, batch_idx):
    batch_images = batch[0]
    # Get the reconstructed images
    reconstructed_images = self.forward(batch_images)
    # Calculate loss
    batch_loss = F.mse_loss(reconstructed_images, batch_images)

    # store the result
    result = pl.EvalResult(checkpoint_on=batch_loss)
    result.batch_loss = batch_loss
    
    return result

  def test_step(self, batch, batch_idx):
    batch_images = batch[0]
    # Get the reconstructed images
    reconstructed_images = self.forward(batch_images)
    # Calculate loss
    batch_loss = F.mse_loss(reconstructed_images, batch_images)

    # store the result
    result = pl.EvalResult(checkpoint_on=batch_loss)
    result.batch_loss = batch_loss
    
    return result   

  def validation_end(self, outputs):
    # Take mean of all batch losses
    avg_loss = outputs.batch_loss.mean()
    result = pl.EvalResult(checkpoint_on=avg_loss)
    result.log('val_loss', avg_loss, prog_bar=True)
    return result

  def test_epoch_end(self, outputs):
    # Take mean of all batch losses
    avg_loss = outputs.batch_loss.mean()
    result = pl.EvalResult()
    result.log('test_loss', avg_loss, prog_bar=True)
    return result    

  def configure_optimizers(self):
    return optim.Adam(self.parameters())

mnist_dm = MNISTDataModule()
model = SimpleAutoEncoder(input_shape=mnist_dm.size(), representation_size=128)
# We use 16-bit precision for lesser memory usage.
# progress_bar_refresh_rate=5, to avoid Colab from crashing
trainer = pl.Trainer(gpus=1, max_epochs=5, precision=16, progress_bar_refresh_rate=5)
trainer.fit(model, mnist_dm)
