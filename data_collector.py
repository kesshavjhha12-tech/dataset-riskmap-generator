"""
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
        # TODO: Replace with actual Sentinel-2 download
        image = np.random.rand(3200, 3200, 3).astype(np.float32)
        np.save(os.path.join(self.aoi_dir, 'raw', 'sentinel.npy'), image)
    
    def collect_population(self):
        print("  Downloading WorldPop...")
        # TODO: Replace with actual WorldPop download
        data = np.random.rand(320, 320).astype(np.float32)
        np.save(os.path.join(self.aoi_dir, 'raw', 'population.npy'), data)
    
    def collect_osm(self):
        print("  Downloading OSM data...")
        # TODO: Replace with actual OSM download
        data = {
            'buildings': np.random.rand(320, 320),
            'roads': np.random.rand(320, 320)
        }
        np.save(os.path.join(self.aoi_dir, 'raw', 'osm.npy'), data)
    
    def collect_landcover(self):
        print("  Downloading ESA WorldCover...")
        # TODO: Replace with actual ESA WorldCover download
        data = np.random.rand(320, 320, 3).astype(np.float32)
        np.save(os.path.join(self.aoi_dir, 'raw', 'landcover.npy'), data)
    
    def collect_nightlights(self):
        print("  Downloading VIIRS...")
        # TODO: Replace with actual VIIRS download
        data = np.random.rand(64, 64).astype(np.float32)
        np.save(os.path.join(self.aoi_dir, 'raw', 'nightlights.npy'), data)
    
    def collect_dem(self):
        print("  Downloading Copernicus DEM...")
        # TODO: Replace with actual DEM download
        data = np.random.rand(320, 320).astype(np.float32)
        np.save(os.path.join(self.aoi_dir, 'raw', 'dem.npy'), data)

def collect_all_aois(aoi_list):
    for aoi in tqdm(aoi_list, desc="Collecting data"):
        DataCollector(aoi).collect_all()