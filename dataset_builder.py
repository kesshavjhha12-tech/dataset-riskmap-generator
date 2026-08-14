"""
Dataset Builder: Final dataset for GCM-HAIRNet training
"""
import os
import numpy as np
import json
from tqdm import tqdm
from scipy.ndimage import zoom
from config import DATA_ROOT

class DatasetBuilder:
    def __init__(self, data_dir=DATA_ROOT):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'processed', 'images')
        self.gis_dir = os.path.join(data_dir, 'processed', 'gis')
        self.label_dir = os.path.join(data_dir, 'processed', 'labels')
        self.metadata_dir = os.path.join(data_dir, 'processed', 'metadata')
        
        for d in [self.image_dir, self.gis_dir, self.label_dir, self.metadata_dir]:
            os.makedirs(d, exist_ok=True)
    
    def process_aoi(self, aoi_name):
        aoi_dir = os.path.join(self.data_dir, aoi_name)
        
        image_path = os.path.join(aoi_dir, 'raw', 'sentinel.npy')
        gis_path = os.path.join(aoi_dir, 'gis', f'{aoi_name}_gis.npy')
        label_path = os.path.join(aoi_dir, 'labels', f'{aoi_name}_risk.npy')
        
        if not all(os.path.exists(p) for p in [image_path, gis_path, label_path]):
            print(f"Missing data for {aoi_name}")
            return
        
        # Load and resize image (3200×3200 → 256×256)
        image = np.load(image_path)
        image = zoom(image, (256/3200, 256/3200, 1))
        
        # Load GIS and label
        gis = np.load(gis_path)
        label = np.load(label_path)
        
        # Save processed data
        np.save(os.path.join(self.image_dir, f'{aoi_name}_image.npy'), image)
        np.save(os.path.join(self.gis_dir, f'{aoi_name}_gis.npy'), gis)
        np.save(os.path.join(self.label_dir, f'{aoi_name}_risk.npy'), label)
        
        # Save metadata
        metadata = {
            'name': aoi_name,
            'image_shape': image.shape,
            'gis_shape': gis.shape,
            'label_shape': label.shape,
            'label_mean': float(label.mean()),
            'label_std': float(label.std()),
            'label_min': float(label.min()),
            'label_max': float(label.max())
        }
        
        with open(os.path.join(self.metadata_dir, f'{aoi_name}_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def build(self, aoi_list):
        print("Building dataset...")
        for aoi in tqdm(aoi_list, desc="Processing AOIs"):
            self.process_aoi(aoi.name)
        self._create_split_file(aoi_list)
        print(f"✅ Dataset ready at {os.path.join(self.data_dir, 'processed')}")
    
    def _create_split_file(self, aoi_list):
        categories = {}
        for aoi in aoi_list:
            if aoi.category not in categories:
                categories[aoi.category] = []
            categories[aoi.category].append(aoi.name)
        
        train_names, val_names, test_names = [], [], []
        
        for category, names in categories.items():
            n = len(names)
            train_names.extend(names[:int(n*0.7)])
            val_names.extend(names[int(n*0.7):int(n*0.85)])
            test_names.extend(names[int(n*0.85):])
        
        splits = {
            'train': train_names,
            'val': val_names,
            'test': test_names
        }
        
        with open(os.path.join(self.data_dir, 'processed', 'splits.json'), 'w') as f:
            json.dump(splits, f, indent=2)