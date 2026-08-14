"""
Save predictions for visualization (run once)
"""
import os
import numpy as np
import torch
import json
from scipy.ndimage import zoom


class UrbanRiskDataset:
    """Dataset that loads and resizes images properly"""
    
    def __init__(self, aoi_names, data_dir='./data'):
        self.aoi_names = aoi_names
        self.data_dir = data_dir
        
        self.samples = []
        for aoi in aoi_names:
            # Try processed folder first (already resized)
            image_path = os.path.join(data_dir, 'processed', 'images', f'{aoi}_image.npy')
            gis_path = os.path.join(data_dir, 'processed', 'gis', f'{aoi}_gis.npy')
            label_path = os.path.join(data_dir, 'processed', 'labels', f'{aoi}_risk.npy')
            
            # Fallback: raw folder with resizing
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
            else:
                print(f"⚠️ Missing data for {aoi}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = np.load(sample['image']).astype(np.float32)
        
        # CRITICAL: Resize to 256x256 if needed
        if image.shape[0] != 256 or image.shape[1] != 256:
            # image is (H, W, C) or (H, W)
            if image.ndim == 3:
                zoom_factors = (256/image.shape[0], 256/image.shape[1], 1)
            else:
                zoom_factors = (256/image.shape[0], 256/image.shape[1])
            image = zoom(image, zoom_factors)
        
        # Load GIS and label
        gis = np.load(sample['gis']).astype(np.float32)
        label = np.load(sample['label']).astype(np.float32)
        
        # Convert to torch tensors (CHW format)
        image = torch.tensor(image).permute(2, 0, 1)  # (3, 256, 256)
        gis = torch.tensor(gis).permute(2, 0, 1)      # (18, 32, 32)
        label = torch.tensor(label).unsqueeze(0)      # (1, 32, 32)
        
        return {
            'image': image,
            'gis': gis,
            'label': label,
            'aoi': sample['aoi']
        }


def save_predictions():
    import torch
    from model import GCMHAIRNet
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = './data'
    checkpoint_dir = './checkpoints'
    
    # Load splits
    splits_path = os.path.join(data_dir, 'processed', 'splits.json')
    
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
    
    print("📊 Loading test dataset...")
    test_dataset = UrbanRiskDataset(splits['test'], data_dir=data_dir)
    print(f"   Found {len(test_dataset)} test samples")
    
    # Load model
    print("📊 Loading model...")
    model = GCMHAIRNet({'device': device}).to(device)
    
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✅ Loaded model from {checkpoint_path}")
    else:
        print(f"   ⚠️ No model found at {checkpoint_path}, using untrained model")
    
    model.eval()
    
    all_images = []
    all_preds = []
    all_targets = []
    all_aois = []
    
    print("📊 Running predictions...")
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            
            # Check image size before feeding to model
            img = sample['image']
            print(f"   Sample {idx}: Image shape = {img.shape}")  # Debug
            
            image = img.unsqueeze(0).to(device)
            gis = sample['gis'].unsqueeze(0).to(device)
            label = sample['label'].unsqueeze(0).to(device)
            
            pred = model(image, gis)
            
            all_images.append(image.cpu().numpy())
            all_preds.append(pred.cpu().numpy())
            all_targets.append(label.cpu().numpy())
            all_aois.append(sample['aoi'])
    
    # Save to disk
    save_dir = './visualization_data'
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, 'images.npy'), np.array(all_images))
    np.save(os.path.join(save_dir, 'preds.npy'), np.array(all_preds))
    np.save(os.path.join(save_dir, 'targets.npy'), np.array(all_targets))
    
    with open(os.path.join(save_dir, 'aois.json'), 'w') as f:
        json.dump(all_aois, f)
    
    print(f"\n✅ Saved predictions to {save_dir}")
    print(f"   Images shape: {np.array(all_images).shape}")
    print(f"   Predictions shape: {np.array(all_preds).shape}")
    print(f"   Targets shape: {np.array(all_targets).shape}")


if __name__ == "__main__":
    # Set environment to CPU if CUDA issues
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    save_predictions()