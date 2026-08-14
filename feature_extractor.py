"""
GIS Feature Extraction: 18 features → 32×32×18 tensor
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import DATA_ROOT, GRID_SIZE, GIS_FEATURES

class FeatureExtractor:
    def __init__(self, aoi, data_dir=DATA_ROOT):
        self.aoi = aoi
        self.data_dir = data_dir
        self.aoi_dir = os.path.join(data_dir, aoi.name)
        
    def extract_all(self):
        print(f"Extracting features for {self.aoi.name}...")
        features = np.zeros((GRID_SIZE, GRID_SIZE, len(GIS_FEATURES)))
        
        for i, feature_name in enumerate(GIS_FEATURES):
            features[:, :, i] = self._extract_feature(feature_name)
        
        # Normalize
        for i in range(features.shape[-1]):
            feat = features[:, :, i]
            min_val, max_val = feat.min(), feat.max()
            if max_val - min_val > 1e-8:
                features[:, :, i] = (feat - min_val) / (max_val - min_val)
        
        self._save_features(features)
        return features
    
    def _extract_feature(self, feature_name):
        """Extract a single feature - placeholder with realistic distributions"""
        x = np.linspace(-1, 1, GRID_SIZE)
        y = np.linspace(-1, 1, GRID_SIZE)
        xx, yy = np.meshgrid(x, y)
        dist = np.sqrt(xx**2 + yy**2)
        
        if 'density' in feature_name or feature_name.endswith('count'):
            # Center-weighted (city center high)
            return np.exp(-dist**2 * 2) * 0.8 + np.random.rand(GRID_SIZE, GRID_SIZE) * 0.2
        elif 'percentage' in feature_name or feature_name in ['ndvi', 'ndbi']:
            # Uniform distribution
            return np.random.rand(GRID_SIZE, GRID_SIZE)
        elif feature_name == 'night_lights':
            # Clustered hotspots
            data = np.zeros((GRID_SIZE, GRID_SIZE))
            for _ in range(3):
                cx, cy = np.random.randint(5, 27, 2)
                xx_shift = (xx + cx/16 - 0.5) * 2
                yy_shift = (yy + cy/16 - 0.5) * 2
                data += np.exp(-(xx_shift**2 + yy_shift**2) * 3)
            return (data - data.min()) / (data.max() - data.min() + 1e-8)
        else:
            return np.random.rand(GRID_SIZE, GRID_SIZE)
    
    def _save_features(self, features):
        # Save as numpy
        np.save(os.path.join(self.aoi_dir, 'gis', f'{self.aoi.name}_gis.npy'), features)
        
        # Save as CSV
        rows = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                row = {
                    'cell_id': i * GRID_SIZE + j,
                    'row': i,
                    'col': j,
                    'latitude': self.aoi.latitude + (i - GRID_SIZE/2) * 0.01,
                    'longitude': self.aoi.longitude + (j - GRID_SIZE/2) * 0.01
                }
                for idx, name in enumerate(GIS_FEATURES):
                    row[name] = features[i, j, idx]
                rows.append(row)
        
        pd.DataFrame(rows).to_csv(os.path.join(self.aoi_dir, f'{self.aoi.name}_features.csv'), index=False)

def extract_all_aois(aoi_list):
    for aoi in tqdm(aoi_list, desc="Extracting features"):
        FeatureExtractor(aoi).extract_all()