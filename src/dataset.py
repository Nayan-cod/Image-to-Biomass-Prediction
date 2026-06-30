import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

# Conversion factor from grams to kg/ha
GRAMS_TO_KG_HA = 47.619

TARGET_NAMES = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
METADATA_COLS = ['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']

def load_and_pivot_metadata(base_dir):
    """
    Loads train.csv and pivots it from long format to wide format.
    """
    csv_path = os.path.join(base_dir, "train.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Pivot the dataframe
    pivoted = df.pivot(index=METADATA_COLS, columns="target_name", values="target").reset_index()
    
    # Ensure all targets are present
    for target in TARGET_NAMES:
        if target not in pivoted.columns:
            pivoted[target] = 0.0
            
    return pivoted

def get_group_split(df, test_size=0.2, random_state=42):
    """
    Performs a group-aware train/validation split based on Sampling_Date.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df['Sampling_Date']))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    return train_df, val_df

class BiomassDataset(Dataset):
    def __init__(self, df, base_dir, img_size=(256, 128), transform=None, is_validation=False):
        """
        PyTorch Dataset for CSIRO Pasture Biomass.
        img_size is (width, height) to match OpenCV resize convention.
        """
        self.df = df
        self.base_dir = base_dir
        self.img_size = img_size
        self.transform = transform
        self.is_validation = is_validation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row['image_path']
        full_path = os.path.join(self.base_dir, rel_path)
        
        # Load image
        img = cv2.imread(full_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {full_path}")
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, self.img_size)
        
        # Normalize image to [0, 1] and transpose to CHW
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        
        # If transform is provided (e.g. data augmentation)
        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        # Extract targets in grams
        targets = row[TARGET_NAMES].values.astype(np.float32)
        targets_tensor = torch.from_numpy(targets)
        
        return img_tensor, targets_tensor

def get_dataloaders(train_df, val_df, base_dir, batch_size=16, img_size=(256, 128)):
    """
    Helper to get training and validation PyTorch DataLoaders.
    """
    train_dataset = BiomassDataset(train_df, base_dir, img_size=img_size)
    val_dataset = BiomassDataset(val_df, base_dir, img_size=img_size, is_validation=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Quick test
    base = r"c:\Users\ratha\OneDrive\Desktop\datavidwan\New folder (2)\csiro-biomass"
    df = load_and_pivot_metadata(base)
    train_df, val_df = get_group_split(df)
    print(f"Total images: {len(df)}")
    print(f"Train images: {len(train_df)}")
    print(f"Val images: {len(val_df)}")
    
    train_loader, val_loader = get_dataloaders(train_df, val_df, base, batch_size=8)
    images, targets = next(iter(train_loader))
    print("Batch images shape:", images.shape)
    print("Batch targets shape:", targets.shape)
