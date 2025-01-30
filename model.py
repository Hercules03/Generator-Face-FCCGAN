import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, latent_dim=100, feature_dim=64):
        super(Generator, self).__init__()
        
        # Initial dense layer
        self.fc = nn.Linear(latent_dim, 4*4*feature_dim*16)
        self.bn_fc = nn.BatchNorm1d(4*4*feature_dim*16)
        
        # Transposed convolution layers
        self.deconv1 = nn.ConvTranspose2d(feature_dim*16, feature_dim*8, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(feature_dim*8)
        
        self.deconv2 = nn.ConvTranspose2d(feature_dim*8, feature_dim*4, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(feature_dim*4)
        
        self.deconv3 = nn.ConvTranspose2d(feature_dim*4, feature_dim*2, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(feature_dim*2)
        
        self.deconv4 = nn.ConvTranspose2d(feature_dim*2, feature_dim, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(feature_dim)
        
        self.deconv5 = nn.ConvTranspose2d(feature_dim, 3, 4, 2, 1)

    def forward(self, x):
        # Dense layer
        x = F.relu(self.bn_fc(self.fc(x)))
        x = x.view(-1, 1024, 4, 4)
        
        # Transposed convolution layers
        x = F.relu(self.bn1(self.deconv1(x)))  # 8x8
        x = F.relu(self.bn2(self.deconv2(x)))  # 16x16
        x = F.relu(self.bn3(self.deconv3(x)))  # 32x32
        x = F.relu(self.bn4(self.deconv4(x)))  # 64x64
        x = torch.tanh(self.deconv5(x))        # 128x128
        
        return x

class Discriminator(nn.Module):
    def __init__(self, feature_dim=64):
        super(Discriminator, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, feature_dim, 4, 2, 1)
        
        self.conv2 = nn.Conv2d(feature_dim, feature_dim*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(feature_dim*2)
        
        self.conv3 = nn.Conv2d(feature_dim*2, feature_dim*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(feature_dim*4)
        
        self.conv4 = nn.Conv2d(feature_dim*4, feature_dim*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(feature_dim*8)
        
        self.conv5 = nn.Conv2d(feature_dim*8, feature_dim*16, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(feature_dim*16)
        
        # Dense layer
        self.fc = nn.Linear(4*4*feature_dim*16, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        
        x = x.view(-1, 4*4*1024)
        x = torch.sigmoid(self.fc(x))
        
        return x

class FFHQGAN:
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()
        
        # Create fixed noise for visualization
        self.fixed_noise = torch.randn(8, self.latent_dim)
    
    def generate(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim)
        if next(self.generator.parameters()).is_cuda:
            z = z.cuda()
        return self.generator(z)

    def get_device(self):
        return next(self.generator.parameters()).device
        
    def plot_epoch_samples(self, epoch):
        try:
            output_dir = "epoch_samples"
            os.makedirs(output_dir, exist_ok=True)
            
            device = self.get_device()
            fixed_noise = self.fixed_noise.to(device)
            
            self.generator.eval()
            with torch.no_grad():
                fake_images = self.generator(fixed_noise)
            self.generator.train()
            
            fake_images = fake_images.cpu()
            
            fig = plt.figure(figsize=(12, 6))
            for i in range(fake_images.size(0)):
                plt.subplot(2, 4, i+1)
                plt.tight_layout()
                
                img = fake_images[i].permute(1, 2, 0)
                img = (img + 1) / 2
                img = torch.clamp(img, 0, 1)
                
                plt.imshow(img)
                plt.title(f"Sample {i+1}")
                plt.axis("off")
            
            save_path = os.path.join(output_dir, f'epoch_{epoch}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error in plot_epoch_samples: {str(e)}")
            import traceback
            traceback.print_exc()

