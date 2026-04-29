import os 
import torch
import pandas as pd

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

dataset = CustomImageDataset(
    annotations_file = "custom_data/label.csv",
    img_dir = "custom_data/images/",
    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])
)

print(len(dataset))

image, label = dataset[0]

print("Image shape:", image.shape)
print("Label:", label)

dataloader = DataLoader(
    dataset,
    batch_size = 2,
    shuffle = True  
)

for X, y in dataloader:
    print("Batch X shape:", X.shape) #[2, 1, 255, 255]
    print("Batch y shape:", y.shape) #[2]
    print("Batch y:", y)
    break
