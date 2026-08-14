"""
AOI Management
"""
from config import AOIS

class AOI:
    def __init__(self, name, lat, lon, category):
        self.name = name
        self.latitude = lat
        self.longitude = lon
        self.category = category
        self.size_km = 32
        self.cell_size_km = 1
        self.grid_size = 32
        self.total_cells = 1024
    
    def __repr__(self):
        return f"AOI(name='{self.name}', category='{self.category}')"

def get_all_aois():
    aois = []
    for name, info in AOIS.items():
        aois.append(AOI(name, info['lat'], info['lon'], info['category']))
    return aois

def get_aoi_names():
    return list(AOIS.keys())