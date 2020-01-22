import glob
from pathlib import Path
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, path_a: str, path_b: str, max_samples_a=100, max_samples_b=1000, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(Path(path_a).rglob('*.*'))[0:max_samples_a]
        self.files_B = sorted(Path(path_b).rglob('*.*'))[0:max_samples_b]

        # Removing some potential junk non-existing db files
        for filepath_a in self.files_A:
            if ".db" in str(filepath_a):
                self.files_A.remove(filepath_a)
        for filepath_b in self.files_B:
            if ".db" in str(filepath_b):
                self.files_B.remove(filepath_b)

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert("RGB"))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert("RGB"))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert("RGB"))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))