import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import os
import numpy as np
import matplotlib as plt

from dataset import FFHQDataset
from model import FFHQGAN


def train_ffhqgan(model, data_path, num_epochs=150, batch_size=32, lr=0.0002, beta1=0.5, beta2=0.999, save_interval=10, image_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.generator.to(device)
    model.discriminator.to(device)

    # Create directories
    os.makedirs('samples', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create dataset and dataloader
    dataset = FFHQDataset(data_path, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        drop_last=True  # Drop the last batch if it's smaller than batch_size
    )

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(model.generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    print(f"Starting Training on {device}...")
    for epoch in range(num_epochs):
        g_losses = []
        d_losses = []
        
        for i, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)
            
            # Skip batches that are too small
            if batch_size < 2:
                continue
                
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            real_images = real_images.to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            d_output_real = model.discriminator(real_images)
            d_loss_real = criterion(d_output_real, real_labels)
            
            z = torch.randn(batch_size, model.latent_dim).to(device)
            fake_images = model.generator(z)
            d_output_fake = model.discriminator(fake_images.detach())
            d_loss_fake = criterion(d_output_fake, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            g_output = model.discriminator(fake_images)
            g_loss = criterion(g_output, real_labels)
            
            g_loss.backward()
            optimizer_G.step()
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            if i % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] '
                      f'D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}')

        # End of epoch processing
        if g_losses and d_losses:  # Check if we have any losses to average
            avg_g_loss = np.mean(g_losses)
            avg_d_loss = np.mean(d_losses)
            print(f'Epoch [{epoch}/{num_epochs}] Average losses - D: {avg_d_loss:.4f} G: {avg_g_loss:.4f}')

            # Plot samples
            model.plot_epoch_samples(epoch)

            # Save checkpoints
            if epoch % save_interval == 0:
                with torch.no_grad():
                    fake_images = model.generator(torch.randn(16, model.latent_dim).to(device))
                    save_image(fake_images.data[:16], f'samples/fake_images_epoch_{epoch}.png', 
                             normalize=True, nrow=4)

                torch.save(model.generator.state_dict(), f'models/generator_epoch_{epoch}.pth')
                torch.save(model.discriminator.state_dict(), f'models/discriminator_epoch_{epoch}.pth')

    # Save final model
    torch.save(model.generator.state_dict(), 'models/generator_final.pth')
    return model

# Configuration
config = {
    'num_epochs': 150,
    'batch_size': 32,
    'learning_rate': 0.0002,
    'beta1': 0.5,
    'beta2': 0.999,
    'latent_dim': 100,
    'save_interval': 10,
    'image_size': 128  # Reduced from 1024 for practical training
}

if __name__ == '__main__':
    # Get the dataset path from kagglehub
    import kagglehub
    data_path = kagglehub.dataset_download("arnaud58/flickrfaceshq-dataset-ffhq")
    
    # Initialize and train the model
    ffhqgan = FFHQGAN(latent_dim=config['latent_dim'])
    trained_model = train_ffhqgan(
        model=ffhqgan,
        data_path=data_path,
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
        beta1=config['beta1'],
        beta2=config['beta2'],
        save_interval=config['save_interval'],
        image_size=config['image_size']
    )