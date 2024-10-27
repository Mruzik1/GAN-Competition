from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os


class MonetDataset(Dataset):
    def __init__(self, root_X: str, root_Y: str, transform=None):
        # orig -> styled
        self.root_X = root_X
        # styled -> orig
        self.root_Y = root_Y
        self.transform = transform

        self.files_X = sorted(os.listdir(root_X))
        self.files_Y = sorted(os.listdir(root_Y))

    def __len__(self) -> int:
        return max(len(self.files_X), len(self.files_Y))

    def __getitem__(self, idx: int) -> tuple:
        img_X = Image.open(os.path.join(self.root_X, self.files_X[idx % len(self.files_X)])).convert("RGB")
        img_Y = Image.open(os.path.join(self.root_Y, self.files_Y[idx % len(self.files_Y)])).convert("RGB")

        if self.transform:
            img_X = self.transform(img_X)
            img_Y = self.transform(img_Y)

        return img_X, img_Y


class MonetDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32, img_size: int = 256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transforms_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def setup(self, stage: str = None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = MonetDataset(
                root_X=os.path.join(self.data_dir, "train_photos"),
                root_Y=os.path.join(self.data_dir, "train_monet"),
                transform=self.transforms_train
            )

        if stage == 'test' or stage is None:
            self.test_dataset = MonetDataset(
                root_X=os.path.join(self.data_dir, "test_photos"),
                root_Y=os.path.join(self.data_dir, "test_monet"),
                transform=self.transforms_test
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4, 
            persistent_workers=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4, 
            persistent_workers=True
        )


if __name__ == "__main__":
    data_module = MonetDataModule()
    data_module.setup("test")
    data_loader = data_module.test_dataloader()
    
    fig, ax = plt.subplots()
    print(next(iter(data_loader))[1][0].transpose(0, 2))
    ax.imshow((next(iter(data_loader))[1][0].transpose(0, 2) + 1) / 2)
    plt.show()