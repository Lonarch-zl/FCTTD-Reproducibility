from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

class MirroredCIFAR10(Dataset):

    def __init__(self, path, transform):
        obj = np.load(path)
        self.data = obj['images']
        self.targets = obj['labels'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        x = Image.fromarray(self.data[i])
        return (self.transform(x) if self.transform else x, self.targets[i])

def get_cifar10(root, train, transform, download=False):
    path = Path(root) / ('train.npz' if train else 'test.npz')
    if path.exists():
        return MirroredCIFAR10(path, transform)
    return CIFAR10(root, train=train, transform=transform, download=download)
