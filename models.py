import torch
from torch import nn
from torch import optim

from pytorch_lightning import LightningModule


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=True):
        super(Upsample, self).__init__()
        self.dropout = dropout
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=nn.InstanceNorm2d),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout_layer = nn.Dropout2d(0.5)

    def forward(self, x, shortcut=None):
        x = self.block(x)
        if self.dropout:
            x = self.dropout_layer(x)

        if shortcut is not None:
            x = torch.cat([x, shortcut], dim=1)

        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, apply_instancenorm=True):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=nn.InstanceNorm2d)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.apply_norm = apply_instancenorm

    def forward(self, x):
        x = self.conv(x)
        if self.apply_norm:
            x = self.norm(x)
        x = self.relu(x)

        return x


class GeneratorGAN(nn.Module):
    def __init__(self, filter=64):
        super(GeneratorGAN, self).__init__()
        self.downsamples = nn.ModuleList([
            Downsample(3, filter, kernel_size=4, apply_instancenorm=False),  # (b, filter, 128, 128)
            Downsample(filter, filter * 2),  # (b, filter * 2, 64, 64)
            Downsample(filter * 2, filter * 4),  # (b, filter * 4, 32, 32)
            Downsample(filter * 4, filter * 8),  # (b, filter * 8, 16, 16)
            Downsample(filter * 8, filter * 8), # (b, filter * 8, 8, 8)
            Downsample(filter * 8, filter * 8), # (b, filter * 8, 4, 4)
            Downsample(filter * 8, filter * 8), # (b, filter * 8, 2, 2)
        ])

        self.upsamples = nn.ModuleList([
            Upsample(filter * 8, filter * 8),
            Upsample(filter * 16, filter * 8),
            Upsample(filter * 16, filter * 8),
            Upsample(filter * 16, filter * 4, dropout=False),
            Upsample(filter * 8, filter * 2, dropout=False),
            Upsample(filter * 4, filter, dropout=False)
        ])

        self.last = nn.Sequential(
            nn.ConvTranspose2d(filter * 2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        skips = []
        for l in self.downsamples:
            x = l(x)
            skips.append(x)

        skips = reversed(skips[:-1])
        for l, s in zip(self.upsamples, skips):
            x = l(x, s)

        out = self.last(x)

        return out


class DiscriminatorPatchGAN(nn.Module):
    def __init__(self, filter=64):
        super(DiscriminatorPatchGAN, self).__init__()

        self.block = nn.Sequential(
            Downsample(3, filter, kernel_size=4, stride=2, apply_instancenorm=False),
            Downsample(filter, filter * 2, kernel_size=4, stride=2),
            Downsample(filter * 2, filter * 4, kernel_size=4, stride=2),
            Downsample(filter * 4, filter * 8, kernel_size=4, stride=1),
        )

        self.last = nn.Conv2d(filter * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.block(x)
        x = self.last(x)

        return x


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()
        
#         # no bias cause we have batch norm
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
        
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
        
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
    
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
        
#         out = self.conv2(out)
#         out = self.bn2(out)
        
#         out += self.shortcut(x)
#         out = self.relu(out)
        
#         return out
    

# class GeneratorGAN(nn.Module):
#     def __init__(self):
#         super(GeneratorGAN, self).__init__()

#         self.relu = nn.ReLU(inplace=True)
#         self.tanh = nn.Tanh()
#         # self.max_pool = nn.MaxPool2d(kernel_size=2)

#         # encoder
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2)

#         # transformation with res blocks
#         self.res1 = ResidualBlock(256, 256, stride=1)
#         self.res2 = ResidualBlock(256, 256, stride=1)
#         self.res3 = ResidualBlock(256, 256, stride=1)
#         self.res4 = ResidualBlock(256, 256, stride=1)
        
#         # decoder
#         self.trans_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1)
#         self.trans_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1)
#         self.conv4 = nn.Conv2d(64, 3, kernel_size=7, stride=1)
    
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))

#         x = self.res1(x)
#         x = self.res2(x)
#         x = self.res3(x)
        
#         x = self.relu(self.trans_conv1(x))
#         x = self.relu(self.trans_conv2(x))
#         x = self.tanh(self.conv4(x))

#         return x
    

# class DiscriminatorPatchGAN(nn.Module):
#     def __init__(self):
#         super(DiscriminatorPatchGAN, self).__init__()

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=nn.InstanceNorm2d)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=nn.InstanceNorm2d)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=2, bias=nn.InstanceNorm2d)
#         self.conv4 = nn.Conv2d(256, 1, kernel_size=4, stride=1)

#         self.sigmoid = nn.Sigmoid()
#         self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

#         self.instance_norm1 = nn.InstanceNorm2d(64)
#         self.instance_norm2 = nn.InstanceNorm2d(128)
#         self.instance_norm3 = nn.InstanceNorm2d(256)

#     def forward(self, x):
#         x = self.leaky_relu(self.instance_norm1(self.conv1(x)))
#         x = self.leaky_relu(self.instance_norm2(self.conv2(x)))
#         x = self.leaky_relu(self.instance_norm3(self.conv3(x)))
#         x = self.sigmoid(self.conv4(x))
#         return x


class CycleGAN(LightningModule):
    def __init__(self, gx, gy, dx, dy, g_lr, d_lr, rec_coef=10, id_coef=2):
        super(CycleGAN, self).__init__()

        # gx: X -> Y, gy: Y -> X, dx(X), dy(Y)
        self.gx = gx
        self.gy = gy
        self.dx = dx
        self.dy = dy

        self.g_lr = g_lr
        self.d_lr = d_lr
        self.rec_coef = rec_coef
        self.id_coef = id_coef

        self.train_step = 0
        self.counter_gen_dis = 0            # to switch between training a generator and a discriminator
        self.automatic_optimization = False
    
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def forward(self, x):
        return self.gx(x)
    
    def training_step(self, batch, batch_idx):
        opt_gx, opt_gy, opt_dx, opt_dy = self.optimizers()
        real_x, real_y = batch

        self.train_step += 1
        self.counter_gen_dis += 1
        if self.counter_gen_dis > 1:
            self.counter_gen_dis = 0

        fake_y = self.gx(real_x)
        fake_x = self.gy(real_y)

        # save images
        if self.train_step % 300 == 0:
            imgs = [
                real_x[0].cpu().detach().permute(1, 2, 0).numpy(), 
                real_y[0].cpu().detach().permute(1, 2, 0).numpy(), 
                fake_y[0].cpu().detach().permute(1, 2, 0).numpy(),
                fake_x[0].cpu().detach().permute(1, 2, 0).numpy(),
            ]
            img_names = [f"real_x", f"real_y", f"fake_x", f"fake_y"]
            self.logger.log_image(key=f"sample", caption=img_names, images=imgs)

        # update generators
        if self.counter_gen_dis >= 1:
            opt_gx.zero_grad()
            opt_gy.zero_grad()
            rec_x = self.gy(fake_y)
            rec_y = self.gx(fake_x)
            fake_dx = self.dx(fake_x).detach()
            fake_dy = self.dy(fake_y).detach()

            val_gx = self.mse(fake_dx, torch.ones_like(fake_dx).to(self.device))
            val_gy = self.mse(fake_dy, torch.ones_like(fake_dy).to(self.device))
            val_loss = (val_gx + val_gy) / 2

            rec_x_loss = self.mae(rec_x, real_x)
            rec_y_loss = self.mae(rec_y, real_y)
            rec_loss = (rec_x_loss + rec_y_loss) / 2

            id_x = self.mae(fake_x, real_x)
            id_y = self.mae(fake_y, real_y)
            id_loss = (id_x + id_y) / 2

            loss_g = val_loss + self.rec_coef*rec_loss + self.id_coef*id_loss

            self.log("loss_g", loss_g)
            self.log("validity", val_loss)
            self.log("recon", rec_loss)
            self.log("identity", id_loss)

            # needs to be done maually now :|
            self.manual_backward(loss_g)
            opt_gx.step()
            opt_gy.step()

        # update discriminators
        else:
            opt_dx.zero_grad()
            opt_dy.zero_grad()
            real_dx = self.dx(real_x)
            real_dy = self.dy(real_y)
            fake_dx = self.dx(fake_x.detach())
            fake_dy = self.dy(fake_y.detach())

            rec_dx_loss = self.mse(fake_dx, torch.zeros_like(fake_dx).to(self.device))
            rec_dy_loss = self.mse(fake_dy, torch.zeros_like(fake_dy).to(self.device))
            rec_loss = (rec_dx_loss + rec_dy_loss) / 2

            val_dx_loss = self.mse(real_dx, torch.ones_like(real_dx).to(self.device))
            val_dy_loss = self.mse(real_dy, torch.ones_like(real_dy).to(self.device))

            loss_d = (rec_loss + val_dx_loss + val_dy_loss) / 3
            self.log("loss_d", loss_d)

            self.manual_backward(loss_d)
            opt_dx.step()
            opt_dy.step()

    def configure_optimizers(self):
        self.gx_optimizer = optim.Adam(self.gx.parameters(), lr=self.g_lr, betas=(0.5, 0.999))
        self.gy_optimizer = optim.Adam(self.gy.parameters(), lr=self.g_lr, betas=(0.5, 0.999))
        self.dx_optimizer = optim.Adam(self.dx.parameters(), lr=self.d_lr, betas=(0.5, 0.999))
        self.dy_optimizer = optim.Adam(self.dy.parameters(), lr=self.d_lr, betas=(0.5, 0.999))

        return [self.gx_optimizer, self.gy_optimizer, self.dx_optimizer, self.dy_optimizer], []


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    generator = GeneratorGAN().to(device)
    discriminator = DiscriminatorPatchGAN().to(device)
    dummy_data = torch.rand((32, 3, 256, 256)).to(device) * 2 - 1

    gen_out = generator(dummy_data)
    disc_out = discriminator(gen_out)

    print(disc_out.shape)