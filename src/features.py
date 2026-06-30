import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
from src.dataset import load_and_pivot_metadata, TARGET_NAMES

def extract_image_features(img_rgb, mask, gray_img):
    """
    Extracts features from an RGB image, its segmentation mask, and its grayscale version.
    mask: 0=BG, 1=Green, 2=Dead
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Vegetation masks
    veg_mask = (mask > 0)
    green_mask = (mask == 1)
    dead_mask = (mask == 2)
    
    total_pixels = mask.size
    veg_pixels = np.sum(veg_mask)
    
    # 1. Vegetation Coverage & Ratios
    veg_coverage = veg_pixels / total_pixels if total_pixels > 0 else 0.0
    green_ratio = np.sum(green_mask) / veg_pixels if veg_pixels > 0 else 0.0
    dead_ratio = np.sum(dead_mask) / veg_pixels if veg_pixels > 0 else 0.0
    
    # 2. Color Stats within Vegetation Mask
    features = {
        'veg_coverage': veg_coverage,
        'green_ratio': green_ratio,
        'dead_ratio': dead_ratio,
    }
    
    channels_rgb = ['R', 'G', 'B']
    channels_hsv = ['H', 'S', 'V']
    
    for i, ch in enumerate(channels_rgb):
        vals = img_rgb[:,:,i][veg_mask]
        features[f'mean_{ch}'] = np.mean(vals) if len(vals) > 0 else 0.0
        features[f'std_{ch}'] = np.std(vals) if len(vals) > 0 else 0.0
        
    for i, ch in enumerate(channels_hsv):
        vals = hsv[:,:,i][veg_mask]
        features[f'mean_{ch}'] = np.mean(vals) if len(vals) > 0 else 0.0
        features[f'std_{ch}'] = np.std(vals) if len(vals) > 0 else 0.0
        
    # 3. Texture Features
    # GLCM (Gray-Level Co-occurrence Matrix)
    # Downsample gray image slightly for GLCM to speed up computation if needed, but 256x128 is already small.
    glcm = graycomatrix(gray_img, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    features['glcm_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
    features['glcm_homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
    features['glcm_energy'] = graycoprops(glcm, 'energy')[0, 0]
    features['glcm_correlation'] = graycoprops(glcm, 'correlation')[0, 0]
    
    # Laplacian Variance (Blurriness/Texture roughness proxy)
    features['laplacian_var'] = cv2.Laplacian(gray_img, cv2.CV_64F).var()
    
    # 4. Canopy Density (Edge Density within veg mask)
    edges = cv2.Canny(gray_img, 50, 150)
    veg_edges = edges[veg_mask]
    features['edge_density'] = np.sum(veg_edges > 0) / veg_pixels if veg_pixels > 0 else 0.0
    
    return features

def build_features_table(df, base_dir, model=None, img_size=(256, 128)):
    """
    Extracts features for all images in the dataframe and returns a feature DataFrame.
    If model (U-Net) is provided, it uses U-Net for masks; otherwise, it falls back to classical.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is not None:
        model.eval()
        model.to(device)
        
    feature_list = []
    
    print("Extracting features for all images...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        rel_path = row['image_path']
        full_path = os.path.join(base_dir, rel_path)
        
        img = cv2.imread(full_path)
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, img_size)
        gray_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        
        # Segment image
        if model is not None:
            img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                logits = model(img_tensor)
                mask = torch.argmax(logits, dim=1).cpu().numpy()[0]
        else:
            # Fallback to classical segmentation
            from src.segmentation import compute_pseudo_mask
            mask = compute_pseudo_mask(img_resized)
            
        # Extract features
        feats = extract_image_features(img_resized, mask, gray_img)
        
        # Add metadata features
        feats['image_path'] = row['image_path']
        feats['Pre_GSHH_NDVI'] = row['Pre_GSHH_NDVI']
        feats['Height_Ave_cm'] = row['Height_Ave_cm']
        feats['Sampling_Date'] = row['Sampling_Date'] # keep for group splitting
        
        # Add target variables
        for target in TARGET_NAMES:
            feats[target] = row[target]
            
        feature_list.append(feats)
        
    features_df = pd.DataFrame(feature_list)
    return features_df

if __name__ == "__main__":
    base = r"c:\Users\ratha\OneDrive\Desktop\datavidwan\New folder (2)\csiro-biomass"
    df = load_and_pivot_metadata(base)
    
    # Build features using classical fallback for speed in this test
    features_df = build_features_table(df, base, model=None)
    
    print("\nFeature table shape:", features_df.shape)
    
    # Compute correlation matrix
    feature_cols = [c for c in features_df.columns if c not in TARGET_NAMES + ['Sampling_Date']]
    corr = features_df[feature_cols + TARGET_NAMES].corr(numeric_only=True)
    
    print("\nCorrelation between features and Dry_Total_g:")
    print(corr['Dry_Total_g'].sort_values(ascending=False).head(10))
    
    # Save to CSV
    out_path = os.path.join(base, "..", "features_table.csv")
    features_df.to_csv(out_path, index=False)
    print(f"\nFeature table saved to: {out_path}")
