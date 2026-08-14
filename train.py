"""
Training Script for GCM-HAIRNet - Using Normalized Data
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
import warnings
from scipy.ndimage import zoom  # <-- ADD THIS IMPORT
warnings.filterwarnings('ignore')

from model import GCMHAIRNet, GCMHAIRNetLoss
from config import DATA_ROOT


class UrbanRiskDataset(Dataset):
    """Dataset for Urban Risk Mapping - Loads normalized data"""
    
    def __init__(self, aoi_names, data_dir=DATA_ROOT, split='train'):
        self.aoi_names = aoi_names
        self.data_dir = data_dir
        self.split = split
        self.target_size = 256  # SwinV2 expects 256x256
        
        self.samples = []
        for aoi in aoi_names:
            # Try processed folder first (already resized and normalized)
            image_path = os.path.join(data_dir, 'processed', 'images', f'{aoi}_image.npy')
            gis_path = os.path.join(data_dir, 'processed', 'gis', f'{aoi}_gis.npy')
            label_path = os.path.join(data_dir, 'processed', 'labels', f'{aoi}_risk.npy')
            
            # Fallback to raw (will resize on the fly)
            if not os.path.exists(image_path):
                image_path = os.path.join(data_dir, aoi, 'raw', 'sentinel.npy')
            if not os.path.exists(gis_path):
                gis_path = os.path.join(data_dir, aoi, 'gis', f'{aoi}_gis.npy')
            if not os.path.exists(label_path):
                label_path = os.path.join(data_dir, aoi, 'labels', f'{aoi}_risk.npy')
            
            if os.path.exists(image_path) and os.path.exists(gis_path) and os.path.exists(label_path):
                self.samples.append({
                    'image': image_path,
                    'gis': gis_path,
                    'label': label_path,
                    'aoi': aoi
                })
        
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def _load_and_resize_image(self, image_path):
        """Load image and resize to target size (256x256)"""
        image = np.load(image_path).astype(np.float32)
        
        # If image is not 256x256, resize it
        if image.shape[0] != self.target_size or image.shape[1] != self.target_size:
            # Determine zoom factors
            if image.ndim == 3:  # (H, W, C)
                zoom_factors = (self.target_size/image.shape[0], 
                                self.target_size/image.shape[1], 
                                1)
            else:  # (H, W)
                zoom_factors = (self.target_size/image.shape[0], 
                                self.target_size/image.shape[1])
            image = zoom(image, zoom_factors)
        
        # Normalize to [0, 1] if not already
        if image.max() > 1.0:
            image = image / 255.0
        
        return image
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and resize image
        image = self._load_and_resize_image(sample['image'])  # (256, 256, 3) or (256, 256)
        
        # Ensure 3 channels
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=2)
        
        # Load GIS and label
        gis = np.load(sample['gis']).astype(np.float32)  # (32, 32, 18)
        label = np.load(sample['label']).astype(np.float32)  # (32, 32)
        
        # Convert to CHW
        image = torch.tensor(image).permute(2, 0, 1)  # (3, 256, 256)
        gis = torch.tensor(gis).permute(2, 0, 1)      # (18, 32, 32)
        label = torch.tensor(label).unsqueeze(0)      # (1, 32, 32)
        
        return {
            'image': image,
            'gis': gis,
            'label': label,
            'aoi': sample['aoi']
        }


def train_model(config):
    """Train GCM-HAIRNet"""
    
    # Load splits
    splits_path = os.path.join(DATA_ROOT, 'processed', 'splits.json')
    
    if not os.path.exists(splits_path):
        print("⚠️ splits.json not found! Creating default...")
        splits = {
            "train": [
                "Bengaluru_CBD", "Whitefield", "Hyderabad", "Chennai",
                "Pune", "Mumbai", "Delhi", "Ahmedabad",
                "Peenya", "Electronic_City",
                "Mysuru", "Bhubaneswar",
                "Bannerghatta", "Ramanagara"
            ],
            "val": ["Hosur", "Mangalore", "Mandya"],
            "test": ["Chennai_Port", "Hoskote", "Chikkaballapur"]
        }
        os.makedirs(os.path.dirname(splits_path), exist_ok=True)
        with open(splits_path, 'w') as f:
            json.dump(splits, f, indent=2)
    else:
        with open(splits_path, 'r') as f:
            splits = json.load(f)
    
    # Create datasets using normalized data
    train_dataset = UrbanRiskDataset(splits['train'], split='train')
    val_dataset = UrbanRiskDataset(splits['val'], split='val')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # Model
    model = GCMHAIRNet(config).to(config['device'])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = GCMHAIRNetLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Training
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch in pbar:
            image = batch['image'].to(config['device'])
            gis = batch['gis'].to(config['device'])
            label = batch['label'].to(config['device'])
            
            pred = model(image, gis)
            loss, _ = criterion(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                image = batch['image'].to(config['device'])
                gis = batch['gis'].to(config['device'])
                label = batch['label'].to(config['device'])
                
                pred = model(image, gis)
                loss, _ = criterion(pred, label)
                val_loss += loss.item()
        
        # Average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"\nEpoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, os.path.join(config['checkpoint_dir'], 'best_model.pth'))
            print(f"  ✅ Saved best model with val_loss: {val_loss:.4f}")
        
        scheduler.step()
    
    # Save history
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history


def main():
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 16,
        'epochs': 100,
        'lr': 1e-4,
        'checkpoint_dir': './checkpoints'
    }
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    print("=" * 60)
    print("GCM-HAIRNet Training (Normalized Data)")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")
    print("=" * 60)
    
    model, history = train_model(config)
    
    print("\n✅ Training Complete!")
    print(f"Best model saved to: {config['checkpoint_dir']}/best_model.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()