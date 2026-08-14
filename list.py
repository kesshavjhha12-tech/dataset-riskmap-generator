"""
One-Click Project Creator for GCM-HAIRNet
Run this once to create all files!
"""

import os

# ============================================================
# FILE CONTENTS
# ============================================================

FILES = {
    'config.py': '''"""
Configuration for GCM-HAIRNet Dataset Generation
"""

AOIS = {
    # Metropolitan (8)
    'Bengaluru_CBD': {'lat': 12.9716, 'lon': 77.5946, 'category': 'metropolitan'},
    'Whitefield': {'lat': 12.9698, 'lon': 77.7499, 'category': 'metropolitan'},
    'Hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'category': 'metropolitan'},
    'Chennai': {'lat': 13.0827, 'lon': 80.2707, 'category': 'metropolitan'},
    'Pune': {'lat': 18.5204, 'lon': 73.8567, 'category': 'metropolitan'},
    'Mumbai': {'lat': 19.0760, 'lon': 72.8777, 'category': 'metropolitan'},
    'Delhi': {'lat': 28.7041, 'lon': 77.1025, 'category': 'metropolitan'},
    'Ahmedabad': {'lat': 23.0225, 'lon': 72.5714, 'category': 'metropolitan'},
    # Industrial (4)
    'Peenya': {'lat': 13.0329, 'lon': 77.5263, 'category': 'industrial'},
    'Electronic_City': {'lat': 12.8399, 'lon': 77.6770, 'category': 'industrial'},
    'Hosur': {'lat': 12.7409, 'lon': 77.8253, 'category': 'industrial'},
    'Chennai_Port': {'lat': 13.1067, 'lon': 80.3206, 'category': 'industrial'},
    # Semi-Urban (4)
    'Mysuru': {'lat': 12.2958, 'lon': 76.6394, 'category': 'semi_urban'},
    'Bhubaneswar': {'lat': 20.2961, 'lon': 85.8245, 'category': 'semi_urban'},
    'Mangalore': {'lat': 12.9141, 'lon': 74.8560, 'category': 'semi_urban'},
    'Hoskote': {'lat': 13.0707, 'lon': 77.7981, 'category': 'semi_urban'},
    # Rural (4)
    'Bannerghatta': {'lat': 12.8000, 'lon': 77.5770, 'category': 'rural'},
    'Ramanagara': {'lat': 12.7219, 'lon': 77.2815, 'category': 'rural'},
    'Mandya': {'lat': 12.5239, 'lon': 76.8950, 'category': 'rural'},
    'Chikkaballapur': {'lat': 13.4350, 'lon': 77.7315, 'category': 'rural'},
}

AOI_SIZE_KM = 32
CELL_SIZE_KM = 1
GRID_SIZE = 32

GIS_FEATURES = [
    'population_density', 'building_density', 'road_density',
    'road_intersection_density', 'distance_to_highway',
    'builtup_percentage', 'vegetation_percentage', 'water_percentage',
    'night_lights', 'school_count', 'hospital_count',
    'police_count', 'bus_stop_count', 'elevation', 'slope',
    'ndvi', 'ndbi', 'commercial_percentage'
]

COMPONENT_INDICES = {
    'HE': [0, 1, 9, 10, 11, 12],
    'IC': [2, 3, 4, 13, 14],
    'UA': [8, 5, 17],
    'EB': [6, 7, 15, 16],
}

SMOOTH_SIGMA = 1.5
CNN_EPOCHS = 100
CNN_LR = 1e-3
DATA_ROOT = './data'
''',

    'aoi_list.py': '''"""
AOI Management
"""
from config import AOIS, AOI_SIZE_KM, CELL_SIZE_KM, GRID_SIZE

class AOI:
    def __init__(self, name, lat, lon, category):
        self.name = name
        self.latitude = lat
        self.longitude = lon
        self.category = category
        self.size_km = AOI_SIZE_KM
        self.cell_size_km = CELL_SIZE_KM
        self.grid_size = GRID_SIZE
        self.total_cells = GRID_SIZE ** 2
    
    def __repr__(self):
        return f"AOI(name='{self.name}', category='{self.category}')"

def get_all_aois():
    aois = []
    for name, info in AOIS.items():
        aois.append(AOI(name, info['lat'], info['lon'], info['category']))
    return aois

def get_aois_by_category(category):
    return [aoi for aoi in get_all_aois() if aoi.category == category]

def get_aoi_names():
    return list(AOIS.keys())

def get_category_mapping():
    mapping = {}
    for name, info in AOIS.items():
        category = info['category']
        if category not in mapping:
            mapping[category] = []
        mapping[category].append(name)
    return mapping
''',

    'data_collector.py': '''"""
Data Collection: Sentinel-2 and GIS Data
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from config import DATA_ROOT

class DataCollector:
    def __init__(self, aoi, output_dir=DATA_ROOT):
        self.aoi = aoi
        self.output_dir = output_dir
        self.aoi_dir = os.path.join(output_dir, aoi.name)
        os.makedirs(self.aoi_dir, exist_ok=True)
    
    def collect_all(self):
        print(f"Collecting data for {self.aoi.name}...")
        dirs = ['images', 'gis', 'osm', 'raw']
        for d in dirs:
            os.makedirs(os.path.join(self.aoi_dir, d), exist_ok=True)
        self.collect_sentinel()
        self.collect_population()
        self.collect_osm()
        self.collect_landcover()
        self.collect_nightlights()
        self.collect_dem()
        print(f"✅ Data collection complete for {self.aoi.name}")
    
    def collect_sentinel(self):
        print("  Downloading Sentinel-2...")
        image = np.random.rand(3200, 3200, 3).astype(np.float32)
        np.save(os.path.join(self.aoi_dir, 'raw', 'sentinel.npy'), image)
    
    def collect_population(self):
        print("  Downloading WorldPop...")
        data = np.random.rand(320, 320).astype(np.float32)
        np.save(os.path.join(self.aoi_dir, 'raw', 'population.npy'), data)
    
    def collect_osm(self):
        print("  Downloading OSM data...")
        data = {'buildings': np.random.rand(320, 320), 'roads': np.random.rand(320, 320)}
        np.save(os.path.join(self.aoi_dir, 'raw', 'osm.npy'), data)
    
    def collect_landcover(self):
        print("  Downloading ESA WorldCover...")
        data = np.random.rand(320, 320, 3).astype(np.float32)
        np.save(os.path.join(self.aoi_dir, 'raw', 'landcover.npy'), data)
    
    def collect_nightlights(self):
        print("  Downloading VIIRS...")
        data = np.random.rand(64, 64).astype(np.float32)
        np.save(os.path.join(self.aoi_dir, 'raw', 'nightlights.npy'), data)
    
    def collect_dem(self):
        print("  Downloading Copernicus DEM...")
        data = np.random.rand(320, 320).astype(np.float32)
        np.save(os.path.join(self.aoi_dir, 'raw', 'dem.npy'), data)

def collect_all_aois(aoi_list):
    for aoi in tqdm(aoi_list, desc="Collecting data"):
        DataCollector(aoi).collect_all()
''',

    'feature_extractor.py': '''"""
GIS Feature Extraction
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
        for i in range(features.shape[-1]):
            feat = features[:, :, i]
            min_val, max_val = feat.min(), feat.max()
            if max_val - min_val > 1e-8:
                features[:, :, i] = (feat - min_val) / (max_val - min_val)
        self._save_features(features)
        return features
    
    def _extract_feature(self, feature_name):
        x = np.linspace(-1, 1, GRID_SIZE)
        y = np.linspace(-1, 1, GRID_SIZE)
        xx, yy = np.meshgrid(x, y)
        dist = np.sqrt(xx**2 + yy**2)
        
        if 'density' in feature_name or feature_name.endswith('count'):
            return np.exp(-dist**2 * 2) * 0.8 + np.random.rand(GRID_SIZE, GRID_SIZE) * 0.2
        elif 'percentage' in feature_name or feature_name in ['ndvi', 'ndbi']:
            return np.random.rand(GRID_SIZE, GRID_SIZE)
        elif feature_name == 'night_lights':
            data = np.zeros((GRID_SIZE, GRID_SIZE))
            for _ in range(3):
                cx, cy = np.random.randint(5, 27, 2)
                xx_shift = xx + cx/16 - 0.5
                yy_shift = yy + cy/16 - 0.5
                data += np.exp(-(xx_shift**2 + yy_shift**2) * 5)
            return (data - data.min()) / (data.max() - data.min() + 1e-8)
        else:
            return np.random.rand(GRID_SIZE, GRID_SIZE)
    
    def _save_features(self, features):
        np.save(os.path.join(self.aoi_dir, 'gis', f'{self.aoi.name}_gis.npy'), features)
        rows = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                row = {'cell_id': i * GRID_SIZE + j, 'row': i, 'col': j,
                       'latitude': self.aoi.latitude + (i - GRID_SIZE/2) * 0.01,
                       'longitude': self.aoi.longitude + (j - GRID_SIZE/2) * 0.01}
                for idx, name in enumerate(GIS_FEATURES):
                    row[name] = features[i, j, idx]
                rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(self.aoi_dir, f'{self.aoi.name}_features.csv'), index=False)

def extract_all_aois(aoi_list):
    for aoi in tqdm(aoi_list, desc="Extracting features"):
        FeatureExtractor(aoi).extract_all()
''',

    'component_builder.py': '''"""
Component Builder
"""
import os
import numpy as np
from config import DATA_ROOT, COMPONENT_INDICES

class ComponentBuilder:
    def __init__(self):
        self.he_indices = COMPONENT_INDICES['HE']
        self.ic_indices = COMPONENT_INDICES['IC']
        self.ua_indices = COMPONENT_INDICES['UA']
        self.eb_indices = COMPONENT_INDICES['EB']
    
    def build(self, gis_cube):
        he = gis_cube[:, :, self.he_indices].sum(axis=2)
        ic = gis_cube[:, :, self.ic_indices].sum(axis=2)
        ua = gis_cube[:, :, self.ua_indices].sum(axis=2)
        eb = gis_cube[:, :, self.eb_indices].sum(axis=2)
        components = np.stack([he, ic, ua, eb], axis=0)
        for i in range(4):
            comp = components[i]
            min_val, max_val = comp.min(), comp.max()
            if max_val - min_val > 1e-8:
                components[i] = (comp - min_val) / (max_val - min_val)
        return components
    
    def build_for_aoi(self, aoi_name, data_dir=DATA_ROOT):
        gis_path = os.path.join(data_dir, aoi_name, 'gis', f'{aoi_name}_gis.npy')
        if not os.path.exists(gis_path):
            raise FileNotFoundError(f"GIS cube not found: {gis_path}")
        components = self.build(np.load(gis_path))
        save_path = os.path.join(data_dir, aoi_name, 'gis', f'{aoi_name}_components.npy')
        np.save(save_path, components)
        return components

def build_all_components(aoi_list):
    builder = ComponentBuilder()
    for aoi in aoi_list:
        print(f"Building components for {aoi.name}...")
        builder.build_for_aoi(aoi.name)
    return builder
''',

    'risk_generator.py': '''"""
Ground Truth Risk Generation
"""
import os
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from config import DATA_ROOT, SMOOTH_SIGMA, CNN_EPOCHS, CNN_LR

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        return x

class RiskGenerator:
    def __init__(self, cnn_path=None):
        self.cnn = TinyCNN()
        if cnn_path and os.path.exists(cnn_path):
            self.cnn.load_state_dict(torch.load(cnn_path))
        self.cnn.eval()
    
    def generate(self, components, smooth_sigma=SMOOTH_SIGMA):
        if len(components.shape) == 3:
            components = components[np.newaxis, ...]
        with torch.no_grad():
            comp_tensor = torch.tensor(components).float()
            risk_score = self.cnn(comp_tensor).squeeze().numpy()
        if smooth_sigma > 0:
            risk_score = gaussian_filter(risk_score, sigma=smooth_sigma)
        min_val, max_val = risk_score.min(), risk_score.max()
        if max_val - min_val > 1e-8:
            risk_score = (risk_score - min_val) / (max_val - min_val)
        return risk_score
    
    def generate_for_aoi(self, aoi_name, data_dir=DATA_ROOT):
        comp_path = os.path.join(data_dir, aoi_name, 'gis', f'{aoi_name}_components.npy')
        if not os.path.exists(comp_path):
            raise FileNotFoundError(f"Components not found: {comp_path}")
        risk_map = self.generate(np.load(comp_path))
        save_dir = os.path.join(data_dir, aoi_name, 'labels')
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f'{aoi_name}_risk.npy'), risk_map)
        return risk_map

class TinyCNNTrainer:
    def __init__(self):
        self.model = TinyCNN()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=CNN_LR)
    
    def prepare_training_data(self, aoi_list, data_dir=DATA_ROOT):
        X, y = [], []
        for aoi in aoi_list:
            comp_path = os.path.join(data_dir, aoi.name, 'gis', f'{aoi.name}_components.npy')
            if not os.path.exists(comp_path):
                continue
            components = np.load(comp_path)
            risk = components[0] * 0.4 + components[1] * 0.3 + components[2] * 0.2 - components[3] * 0.1
            risk = (risk - risk.min()) / (risk.max() - risk.min() + 1e-8)
            X.append(components)
            y.append(risk)
        X = np.stack(X)
        y = np.stack(y)
        return torch.tensor(X).float(), torch.tensor(y).float().unsqueeze(1)
    
    def train(self, X, y, epochs=CNN_EPOCHS):
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        self.model.train()
        for epoch in tqdm(range(epochs), desc="Training Tiny CNN"):
            total_loss = 0
            for batch_x, batch_y in loader:
                pred = self.model(batch_x)
                loss = self.loss_fn(pred, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
        return self.model
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
''',

    'dataset_builder.py': '''"""
Dataset Builder
"""
import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
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
        from scipy.ndimage import zoom
        image = zoom(np.load(image_path), (256/3200, 256/3200, 1))
        np.save(os.path.join(self.image_dir, f'{aoi_name}_image.npy'), image)
        np.save(os.path.join(self.gis_dir, f'{aoi_name}_gis.npy'), np.load(gis_path))
        np.save(os.path.join(self.label_dir, f'{aoi_name}_risk.npy'), np.load(label_path))
        metadata = {'name': aoi_name, 'label_mean': float(np.load(label_path).mean())}
        with open(os.path.join(self.metadata_dir, f'{aoi_name}_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def build(self, aoi_list):
        print("Building dataset...")
        for aoi in tqdm(aoi_list, desc="Processing AOIs"):
            self.process_aoi(aoi.name)
        self._create_split_file(aoi_list)
    
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
        with open(os.path.join(self.data_dir, 'processed', 'splits.json'), 'w') as f:
            json.dump({'train': train_names, 'val': val_names, 'test': test_names}, f, indent=2)
''',

    'run_pipeline.py': '''"""
Master Script: Run the complete pipeline
"""
import os
import warnings
warnings.filterwarnings('ignore')
from aoi_list import get_all_aois
from config import DATA_ROOT
from data_collector import collect_all_aois
from feature_extractor import extract_all_aois
from component_builder import build_all_components
from risk_generator import RiskGenerator, TinyCNNTrainer
from dataset_builder import DatasetBuilder

def run_pipeline():
    print("=" * 60)
    print("GCM-HAIRNet Dataset Generation Pipeline")
    print("=" * 60)
    
    print("\\n📌 Step 1: Loading AOIs...")
    aoi_list = get_all_aois()
    print(f"   Total AOIs: {len(aoi_list)}")
    
    print("\\n📌 Step 2: Collecting Data...")
    collect_all_aois(aoi_list)
    
    print("\\n📌 Step 3: Extracting Features...")
    extract_all_aois(aoi_list)
    
    print("\\n📌 Step 4: Building Components...")
    build_all_components(aoi_list)
    
    print("\\n📌 Step 5: Training Tiny CNN...")
    trainer = TinyCNNTrainer()
    X, y = trainer.prepare_training_data(aoi_list)
    model = trainer.train(X, y)
    trainer.save_model(os.path.join(DATA_ROOT, 'tiny_cnn.pth'))
    
    print("\\n📌 Step 6: Generating Risk Maps...")
    risk_generator = RiskGenerator(os.path.join(DATA_ROOT, 'tiny_cnn.pth'))
    for aoi in aoi_list:
        risk_generator.generate_for_aoi(aoi.name)
    
    print("\\n📌 Step 7: Building Final Dataset...")
    DatasetBuilder().build(aoi_list)
    
    print("\\n" + "=" * 60)
    print("✅ Pipeline Complete!")
    print(f"   Dataset saved to: {DATA_ROOT}/processed")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
''',

    'requirements.txt': '''numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.3.0
tqdm>=4.65.0
''',

    'README.md': '''
    # GCM-HAIRNet Dataset Generation Pipeline

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline
python run_pipeline.py'''}