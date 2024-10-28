import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from models import GeneratorGAN, DiscriminatorPatchGAN, CycleGAN
from dataloader import MonetDataModule
from utils import init_weights


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    epochs = 30

    print("Initializing generators and discriminators...")
    generator_x = GeneratorGAN().to(device)
    discriminator_x = DiscriminatorPatchGAN().to(device)
    generator_y = GeneratorGAN().to(device)
    discriminator_y = DiscriminatorPatchGAN().to(device)

    for model in [generator_x, generator_y, discriminator_x, discriminator_y]:
        init_weights(model, init_type="xavier")

    print("Initializing CycleGAN...")
    # cycle_gan = CycleGAN.load_from_checkpoint(
    #     "./CycleGAN/nsfrhc0e/checkpoints/epoch=5-step=4224.ckpt",
    #     gx=generator_x, 
    #     gy=generator_y, 
    #     dx=discriminator_x, 
    #     dy=discriminator_y,
    #     g_lr=0.0002,
    #     d_lr=0.0002,
    #     rec_coef=5,
    #     id_coef=2,
    # ).to(device)
    cycle_gan = CycleGAN(
        gx=generator_x, 
        gy=generator_y, 
        dx=discriminator_x, 
        dy=discriminator_y,
        g_lr=0.0002,
        d_lr=0.0002,
        rec_coef=5,
        id_coef=2,
    ).to(device)

    print("Loading data...")
    data_module = MonetDataModule(
        data_dir="./data",
        batch_size=16,
        img_size=256
    )

    print("Loading trainer...")
    trainer = Trainer(
        logger=WandbLogger(project="CycleGAN"),
        max_epochs=epochs,
        num_sanity_val_steps=0
    )

    print("Fitting...")
    trainer.fit(model=cycle_gan, datamodule=data_module)