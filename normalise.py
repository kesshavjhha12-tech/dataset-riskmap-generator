"""
Normalize the entire dataset for GCM-HAIRNet
"""
import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from config import DATA_ROOT, GIS_FEATURES, AOIS


class DatasetNormalizer:
    """Normalize GIS features and images for training"""
    
    def __init__(self, data_dir=DATA_ROOT):
        self.data_dir = data_dir
        
        # Stats for each feature
        self.feature_stats = {}
        
        # Stats for images
        self.image_stats = {}
    
    def compute_feature_stats(self, aoi_list):
        """Compute min/max for each feature across all AOIs"""
        print("📊 Computing feature statistics...")
        
        # Initialize stats
        for feature in GIS_FEATURES:
            self.feature_stats[feature] = {'min': float('inf'), 'max': float('-inf')}
        
        # Scan all AOIs
        for aoi_name in tqdm(aoi_list, desc="Scanning AOIs"):
            gis_path = os.path.join(self.data_dir, aoi_name, 'gis', f'{aoi_name}_gis.npy')
            if os.path.exists(gis_path):
                gis = np.load(gis_path)  # (32, 32, 18)
                
                for i, feature in enumerate(GIS_FEATURES):
                    feature_data = gis[:, :, i]
                    if feature_data.min() < self.feature_stats[feature]['min']:
                        self.feature_stats[feature]['min'] = feature_data.min()
                    if feature_data.max() > self.feature_stats[feature]['max']:
                        self.feature_stats[feature]['max'] = feature_data.max()
        
        # Save stats
        stats_path = os.path.join(self.data_dir, 'normalization_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.feature_stats, f, indent=2)
        
        print(f"✅ Feature stats saved to: {stats_path}")
        return self.feature_stats
    
    def compute_image_stats(self, aoi_list):
        """Compute mean and std for images"""
        print("📊 Computing image statistics...")
        
        all_images = []
        for aoi_name in tqdm(aoi_list, desc="Scanning images"):
            image_path = os.path.join(self.data_dir, aoi_name, 'raw', 'sentinel.npy')
            if os.path.exists(image_path):
                image = np.load(image_path)  # (3200, 3200, 3)
                all_images.append(image)
        
        if all_images:
            all_images = np.concatenate([img.reshape(-1, 3) for img in all_images], axis=0)
            
            self.image_stats = {
                'mean': all_images.mean(axis=0).tolist(),
                'std': all_images.std(axis=0).tolist()
            }
            
            stats_path = os.path.join(self.data_dir, 'image_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(self.image_stats, f, indent=2)
            
            print(f"✅ Image stats saved to: {stats_path}")
        
        return self.image_stats
    
    def normalize_aoi(self, aoi_name, stats):
        """Normalize a single AOI"""
        
        # Load GIS
        gis_path = os.path.join(self.data_dir, aoi_name, 'gis', f'{aoi_name}_gis.npy')
        if not os.path.exists(gis_path):
            print(f"⚠️ GIS not found for {aoi_name}")
            return None
        
        gis = np.load(gis_path)  # (32, 32, 18)
        gis_normalized = np.zeros_like(gis)
        
        # Normalize each feature
        for i, feature in enumerate(GIS_FEATURES):
            feature_data = gis[:, :, i]
            min_val = stats[feature]['min']
            max_val = stats[feature]['max']
            
            if max_val - min_val > 1e-8:
                gis_normalized[:, :, i] = (feature_data - min_val) / (max_val - min_val)
            else:
                gis_normalized[:, :, i] = feature_data
        
        # Save normalized GIS
        norm_gis_path = os.path.join(self.data_dir, aoi_name, 'gis', f'{aoi_name}_gis_normalized.npy')
        np.save(norm_gis_path, gis_normalized)
        
        # Update features.csv
        csv_path = os.path.join(self.data_dir, aoi_name, f'{aoi_name}_features.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for i, feature in enumerate(GIS_FEATURES):
                df[feature] = gis_normalized[:, :, i].flatten()
            df.to_csv(os.path.join(self.data_dir, aoi_name, f'{aoi_name}_features_normalized.csv'), index=False)
            print(f"✅ Normalized CSV saved for {aoi_name}")
        
        return gis_normalized
    
    def normalize_images(self, aoi_list):
        """Normalize images to [0, 1]"""
        
        for aoi_name in tqdm(aoi_list, desc="Normalizing images"):
            image_path = os.path.join(self.data_dir, aoi_name, 'raw', 'sentinel.npy')
            if os.path.exists(image_path):
                image = np.load(image_path)
                image = image / 255.0
                np.save(os.path.join(self.data_dir, aoi_name, 'images', f'{aoi_name}_image_normalized.npy'), image)
    
    def process_all_aois(self, aoi_list):
        """Complete normalization pipeline"""
        
        # Step 1: Compute stats
        feature_stats = self.compute_feature_stats(aoi_list)
        
        # Step 2: Normalize each AOI
        print("\n📊 Normalizing GIS features...")
        for aoi_name in tqdm(aoi_list, desc="Normalizing"):
            self.normalize_aoi(aoi_name, feature_stats)
        
        # Step 3: Normalize images
        print("\n📊 Normalizing images...")
        self.normalize_images(aoi_list)
        
        print("\n✅ Dataset normalization complete!")


def create_normalized_processed_dataset():
    """Create processed dataset from normalized files"""
    
    processed_dir = os.path.join(DATA_ROOT, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Create subdirectories
    for subdir in ['images', 'gis', 'labels']:
        os.makedirs(os.path.join(processed_dir, subdir), exist_ok=True)
    
    for aoi_name in tqdm(AOIS.keys(), desc="Creating processed dataset"):
        # Load normalized GIS
        gis_path = os.path.join(DATA_ROOT, aoi_name, 'gis', f'{aoi_name}_gis_normalized.npy')
        if os.path.exists(gis_path):
            gis = np.load(gis_path)
        else:
            # Fallback to original
            gis_path = os.path.join(DATA_ROOT, aoi_name, 'gis', f'{aoi_name}_gis.npy')
            if os.path.exists(gis_path):
                gis = np.load(gis_path)
            else:
                continue
        
        # Load normalized image
        image_path = os.path.join(DATA_ROOT, aoi_name, 'images', f'{aoi_name}_image_normalized.npy')
        if os.path.exists(image_path):
            image = np.load(image_path)
        else:
            # Fallback to resized original
            from scipy.ndimage import zoom
            sentinel_path = os.path.join(DATA_ROOT, aoi_name, 'raw', 'sentinel.npy')
            if os.path.exists(sentinel_path):
                image = zoom(np.load(sentinel_path), (256/3200, 256/3200, 1))
            else:
                continue
        
        # Load risk (already normalized)
        risk_path = os.path.join(DATA_ROOT, aoi_name, 'labels', f'{aoi_name}_risk.npy')
        if os.path.exists(risk_path):
            risk = np.load(risk_path)
        else:
            continue
        
        # Save
        np.save(os.path.join(processed_dir, 'images', f'{aoi_name}_image.npy'), image)
        np.save(os.path.join(processed_dir, 'gis', f'{aoi_name}_gis.npy'), gis)
        np.save(os.path.join(processed_dir, 'labels', f'{aoi_name}_risk.npy'), risk)
    
    print(f"✅ Processed dataset saved to: {processed_dir}")


if __name__ == "__main__":
    from aoi_list import get_all_aois
    
    aoi_list = get_all_aois()
    aoi_names = [aoi.name for aoi in aoi_list]
    
    print("=" * 60)
    print("GCM-HAIRNet Dataset Normalization")
    print("=" * 60)
    print(f"AOIs: {len(aoi_names)}")
    print("=" * 60)
    
    # Initialize normalizer
    normalizer = DatasetNormalizer()
    
    # Process all AOIs
    normalizer.process_all_aois(aoi_names)
    
    # Create processed dataset
    print("\n📊 Creating processed dataset...")
    create_normalized_processed_dataset()
    
    print("\n" + "=" * 60)
    print("✅ All done!")
    print(f"   Dataset ready at: {DATA_ROOT}/processed")
    print("=" * 60)