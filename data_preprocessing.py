import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

class Preprocessor:
    def __init__(self, client_id, data_dir='data', output_dir='preprocessed_data', img_size=224):
        self.client_id = client_id
        self.input_path = Path(data_dir) / client_id
        self.output_path = Path(output_dir) / client_id
        self.img_size = img_size
        self.classes = ['scol', 'norm']
        self.data = []

    def anisotropic_diffusion(self, image, alpha=0.03, kappa=0.1, dt=0.1, num_steps=50):
        image = image.astype('float32')
        for _ in range(num_steps):
            grad_north = np.roll(image, -1, axis=0) - image
            grad_south = np.roll(image, 1, axis=0) - image
            grad_east = np.roll(image, -1, axis=1) - image
            grad_west = np.roll(image, 1, axis=1) - image

            c_north = np.exp(-(grad_north / kappa) ** 2)
            c_south = np.exp(-(grad_south / kappa) ** 2)
            c_east = np.exp(-(grad_east / kappa) ** 2)
            c_west = np.exp(-(grad_west / kappa) ** 2)

            image += dt * (
                c_north * grad_north +
                c_south * grad_south +
                c_east * grad_east +
                c_west * grad_west
            )
        return image

    def load_data(self):
        for cls in self.classes:
            folder = self.input_path / cls
            for img_name in os.listdir(folder):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.data.append((str(folder / img_name), cls))

    def split_data(self):
        train_val, test = train_test_split(
            self.data, test_size=0.15,
            stratify=[label for _, label in self.data],
            random_state=42
        )
        train, val = train_test_split(
            train_val, test_size=0.1765,  # ≈ 15% val
            stratify=[label for _, label in train_val],
            random_state=42
        )
        return {'train': train, 'val': val, 'test': test}

    def process_and_save(self, split_data):
        for split_name, samples in split_data.items():
            for img_path, label in tqdm(samples, desc=f"[{self.client_id}] Processing {split_name}"):
                # Read and preprocess image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = self.anisotropic_diffusion(img)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    
                # Create output path for processed image
                out_dir = self.output_path / split_name / label
                out_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    
            
                out_filename = os.path.basename(img_path)
                out_path = out_dir / out_filename
    
                # Save processed image
                cv2.imwrite(str(out_path), img)


    def run(self):
        print(f"📁 Starting preprocessing for client: {self.client_id}")
        self.load_data()
        split_data = self.split_data()
        self.process_and_save(split_data)
        print(f"✅ Done preprocessing for {self.client_id}.\nSaved to: {self.output_path}\n")


class AISDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=224):
        self.root_dir = Path(root_dir) / split
        self.img_size = img_size
        self.samples = []

        for label in ['scol', 'norm']:
            label_dir = self.root_dir / label
            for img_file in os.listdir(label_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((label_dir / img_file, 0 if label == 'norm' else 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0  # (1, H, W)

        return image, label


def get_data_loaders(client_id, batch_size=32, img_size=224):
    """Return train, validation, and test loaders for a given client."""
    base_path = f"preprocessed_data/{client_id}"

    # Assuming 'train', 'val', and 'test' directories are present in each client folder
    train_dataset = AISDataset(base_path, split='train', img_size=img_size)
    val_dataset = AISDataset(base_path, split='val', img_size=img_size)
    test_dataset = AISDataset(base_path, split='test', img_size=img_size)

    # Create DataLoader for training, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

'''
client_id = "client2"
preprocessor = Preprocessor(client_id)
preprocessor.run()
'''