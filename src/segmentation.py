import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from src.dataset import load_and_pivot_metadata, get_group_split

def compute_pseudo_mask(img_rgb):
    """
    Computes a 3-class pseudo-mask using classical color heuristics.
    0: Background (soil/shadow/other)
    1: Green Vegetation (grass/clover)
    2: Dead/Brown Material
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    R = img_rgb[:,:,0].astype(float)
    G = img_rgb[:,:,1].astype(float)
    B = img_rgb[:,:,2].astype(float)
    
    # Excess Green Index
    exg = 2.0 * G - R - B
    
    # Green heuristic: Hue in [35, 88], Saturation >= 25, Value >= 25, and ExG > 5
    green_mask = (h >= 35) & (h <= 90) & (s >= 25) & (v >= 25) & (exg > 5)
    
    # Dead/yellow heuristic: Hue in [10, 35), Saturation >= 25, Value >= 25, and Red > Blue
    dead_mask = (h >= 10) & (h < 35) & (s >= 25) & (v >= 25) & (R > B * 1.1) & (~green_mask)
    
    # Create mask
    mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    mask[green_mask] = 1
    mask[dead_mask] = 2
    
    return mask

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class LightweightUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up2 = DoubleConv(64, 32)
        
        self.up3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv_up3 = DoubleConv(32, 16)
        
        self.outc = nn.Conv2d(16, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        
        logits = self.outc(x)
        return logits

class WeakSegmentationDataset(Dataset):
    def __init__(self, df, base_dir, img_size=(256, 128)):
        self.df = df
        self.base_dir = base_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        full_path = os.path.join(self.base_dir, row['image_path'])
        
        img = cv2.imread(full_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {full_path}")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.img_size)
        
        # Compute pseudo mask
        pseudo_mask = compute_pseudo_mask(img_resized)
        
        # Convert to tensors
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy(pseudo_mask).long()
        
        return img_tensor, mask_tensor

def train_unet(train_df, base_dir, img_size=(256, 128), epochs=5, batch_size=16):
    """
    Trains a lightweight U-Net on pseudo masks to refine segmentation.
    """
    print(f"Training weak-supervised U-Net on CPU for {epochs} epochs...")
    dataset = WeakSegmentationDataset(train_df, base_dir, img_size=img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightUNet(in_channels=3, out_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * images.size(0)
            
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / len(dataset):.4f}")
        
    return model

def save_segmentation_qa_plots(model, val_df, base_dir, out_dir, img_size=(256, 128), num_samples=4):
    """
    Runs inference on a few validation images and saves comparative plots for QA.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    os.makedirs(out_dir, exist_ok=True)
    samples = val_df.sample(n=min(num_samples, len(val_df)), random_state=42)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)
        
    cmap_mask = plt.cm.get_cmap('viridis', 3)
    
    for idx, (_, row) in enumerate(samples.iterrows()):
        full_path = os.path.join(base_dir, row['image_path'])
        img = cv2.imread(full_path)
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, img_size)
        
        # Classical pseudo-mask
        pseudo = compute_pseudo_mask(img_resized)
        
        # Model predicted mask
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            logits = model(img_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()[0]
            
        # Draw on plot
        axes[idx, 0].imshow(img_resized)
        axes[idx, 0].set_title("Original RGB")
        axes[idx, 0].axis("off")
        
        axes[idx, 1].imshow(pseudo, cmap=cmap_mask, vmin=0, vmax=2)
        axes[idx, 1].set_title("Classical Pseudo Mask")
        axes[idx, 1].axis("off")
        
        axes[idx, 2].imshow(preds, cmap=cmap_mask, vmin=0, vmax=2)
        axes[idx, 2].set_title("U-Net Refined Mask")
        axes[idx, 2].axis("off")
        
    plt.tight_layout()
    out_path = os.path.join(out_dir, "segmentation_qa.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Segmentation QA plot saved to: {out_path}")

if __name__ == "__main__":
    base = r"c:\Users\ratha\OneDrive\Desktop\datavidwan\New folder (2)\csiro-biomass"
    df = load_and_pivot_metadata(base)
    train_df, val_df = get_group_split(df)
    
    # Train model
    unet = train_unet(train_df, base, epochs=2) # 2 epochs for quick verification
    
    # Save QA plots
    out_dir = r"c:\Users\ratha\OneDrive\Desktop\datavidwan\New folder (2)"
    save_segmentation_qa_plots(unet, val_df, base, out_dir)
