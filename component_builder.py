"""
Component Builder: 18 features → 4 components (HE, IC, UA, EB)
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
        
        # Normalize each component
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