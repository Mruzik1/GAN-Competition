import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # no bias cause we have batch norm
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(x)
        out = self.relu(out)
        
        return out
    

class GeneratorGAN(nn.Module):
    def __init__(self):
        super(GeneratorGAN, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        # self.max_pool = nn.MaxPool2d(kernel_size=2)

        # encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=2)

        # transformation with res blocks
        self.res1 = ResidualBlock(128, 128, stride=1)
        self.res2 = ResidualBlock(128, 128, stride=1)
        self.res3 = ResidualBlock(128, 128, stride=1)
        
        # decoder
        self.trans_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1)
        self.trans_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=7, stride=1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        x = self.relu(self.trans_conv1(x))
        x = self.relu(self.trans_conv2(x))
        x = self.tanh(self.conv4(x))

        return x
    

class DiscriminatorPatchGAN(nn.Module):
    def __init__(self):
        super(DiscriminatorPatchGAN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=2)
        self.conv4 = nn.Conv2d(128, 1, kernel_size=4, stride=1)

        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.instance_norm1 = nn.InstanceNorm2d(32)
        self.instance_norm2 = nn.InstanceNorm2d(64)
        self.instance_norm3 = nn.InstanceNorm2d(128)

    def forward(self, x):
        x = self.leaky_relu(self.instance_norm1(self.conv1(x)))
        x = self.leaky_relu(self.instance_norm2(self.conv2(x)))
        x = self.leaky_relu(self.instance_norm3(self.conv3(x)))
        x = self.sigmoid(self.conv4(x))
        return x


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    generator = GeneratorGAN().to(device)
    discriminator = DiscriminatorPatchGAN().to(device)
    dummy_data = torch.rand((32, 3, 256, 256)).to(device) * 2 - 1

    gen_out = generator(dummy_data)
    disc_out = discriminator(gen_out)

    print(disc_out.shape)