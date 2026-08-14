"""
Create QGIS Projects for Each AOI
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from config import DATA_ROOT, AOIS

def create_qgis_project_template(aoi_name):
    """Create a QGIS project template for an AOI"""
    
    # Paths
    aoi_dir = os.path.join(DATA_ROOT, aoi_name)
    qgis_dir = os.path.join(aoi_dir, 'qgis')
    os.makedirs(qgis_dir, exist_ok=True)
    
    # Get AOI info
    aoi_info = AOIS.get(aoi_name, {})
    lat = aoi_info.get('lat', 0)
    lon = aoi_info.get('lon', 0)
    category = aoi_info.get('category', 'unknown')
    
    # 1. Create QGIS Project File (.qgs)
    qgs_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<qgis projectname="{aoi_name}" version="3.28.0">
  <title>{aoi_name} - Urban Risk Assessment</title>
  <properties>
    <ProjectCrs>
      <ID>EPSG:4326</ID>
    </ProjectCrs>
  </properties>
  <legend>
    <legendgroup name="{aoi_name}" open="True">
      <legendgroup name="01 - Satellite Imagery" open="True"/>
      <legendgroup name="02 - GIS Features" open="True"/>
      <legendgroup name="03 - Risk Maps" open="True"/>
      <legendgroup name="04 - Grid" open="True"/>
    </legendgroup>
  </legend>
  <mapcanvas>
    <extent>
      <xmin>{lon - 0.15}</xmin>
      <xmax>{lon + 0.15}</xmax>
      <ymin>{lat - 0.15}</ymin>
      <ymax>{lat + 0.15}</ymax>
    </extent>
  </mapcanvas>
</qgis>'''
    
    qgs_path = os.path.join(qgis_dir, f'{aoi_name}.qgs')
    with open(qgs_path, 'w', encoding='utf-8') as f:
        f.write(qgs_content)
    
    print(f"Created QGIS project: {qgs_path}")
    return qgs_path


def create_geojson_grid(aoi_name):
    """Create GeoJSON grid for an AOI"""
    
    import geopandas as gpd
    from shapely.geometry import Polygon
    
    # Load features CSV
    csv_path = os.path.join(DATA_ROOT, aoi_name, f'{aoi_name}_features.csv')
    
    if not os.path.exists(csv_path):
        print(f"⚠️ CSV not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Get AOI center
    aoi_info = AOIS.get(aoi_name, {})
    lat = aoi_info.get('lat', 12.9716)
    lon = aoi_info.get('lon', 77.5946)
    
    # Create polygons for each cell
    polygons = []
    cell_size = 0.009  # ~1km in degrees
    
    for _, row in df.iterrows():
        row_idx = row['row']
        col_idx = row['col']
        
        # Calculate cell corners
        min_lon = lon - 0.144 + col_idx * cell_size
        max_lon = min_lon + cell_size
        min_lat = lat - 0.144 + row_idx * cell_size
        max_lat = min_lat + cell_size
        
        poly = Polygon([
            (min_lon, min_lat),
            (min_lon, max_lat),
            (max_lon, max_lat),
            (max_lon, min_lat)
        ])
        polygons.append(poly)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=polygons, crs="EPSG:4326")
    
    # Save
    grid_path = os.path.join(DATA_ROOT, aoi_name, 'qgis', f'{aoi_name}_grid.geojson')
    gdf.to_file(grid_path, driver='GeoJSON')
    
    print(f"Created grid GeoJSON: {grid_path}")
    return grid_path


def create_risk_geotiff(aoi_name):
    """Create GeoTIFF from risk map"""
    
    import rasterio
    from rasterio.transform import from_origin
    
    # Load risk map
    risk_path = os.path.join(DATA_ROOT, aoi_name, 'labels', f'{aoi_name}_risk.npy')
    
    if not os.path.exists(risk_path):
        print(f"⚠️ Risk map not found: {risk_path}")
        return None
    
    risk = np.load(risk_path)
    
    # Get AOI center
    aoi_info = AOIS.get(aoi_name, {})
    lat = aoi_info.get('lat', 12.9716)
    lon = aoi_info.get('lon', 77.5946)
    
    # Create GeoTIFF
    tiff_path = os.path.join(DATA_ROOT, aoi_name, 'qgis', f'{aoi_name}_risk.tif')
    
    # Define transform (assuming 1km cells)
    cell_size = 0.009
    transform = from_origin(
        lon - 0.144,  # west
        lat + 0.144,  # north
        cell_size,
        cell_size
    )
    
    with rasterio.open(
        tiff_path,
        'w',
        driver='GTiff',
        height=risk.shape[0],
        width=risk.shape[1],
        count=1,
        dtype=risk.dtype,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(risk, 1)
    
    print(f"Created risk GeoTIFF: {tiff_path}")
    return tiff_path


def create_component_geotiffs(aoi_name):
    """Create GeoTIFFs for each component"""
    
    import rasterio
    from rasterio.transform import from_origin
    
    # Load components
    comp_path = os.path.join(DATA_ROOT, aoi_name, 'gis', f'{aoi_name}_components.npy')
    
    if not os.path.exists(comp_path):
        print(f"⚠️ Components not found: {comp_path}")
        return None
    
    components = np.load(comp_path)  # (4, 32, 32)
    
    # Get AOI center
    aoi_info = AOIS.get(aoi_name, {})
    lat = aoi_info.get('lat', 12.9716)
    lon = aoi_info.get('lon', 77.5946)
    
    component_names = ['Human_Exposure', 'Infrastructure', 'Urban_Activity', 'Environmental_Buffer']
    
    cell_size = 0.009
    transform = from_origin(
        lon - 0.144,
        lat + 0.144,
        cell_size,
        cell_size
    )
    
    for i, name in enumerate(component_names):
        tiff_path = os.path.join(DATA_ROOT, aoi_name, 'qgis', f'{aoi_name}_{name}.tif')
        
        with rasterio.open(
            tiff_path,
            'w',
            driver='GTiff',
            height=components.shape[1],
            width=components.shape[2],
            count=1,
            dtype=components.dtype,
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(components[i], 1)
        
        print(f"✅ Created component GeoTIFF: {tiff_path}")


def create_feature_geotiffs(aoi_name):
    """Create GeoTIFFs for key GIS features"""
    
    import rasterio
    from rasterio.transform import from_origin
    
    # Load GIS features
    gis_path = os.path.join(DATA_ROOT, aoi_name, 'gis', f'{aoi_name}_gis.npy')
    
    if not os.path.exists(gis_path):
        print(f"GIS features not found: {gis_path}")
        return None
    
    gis = np.load(gis_path)  # (32, 32, 18)
    
    # Get AOI center
    aoi_info = AOIS.get(aoi_name, {})
    lat = aoi_info.get('lat', 12.9716)
    lon = aoi_info.get('lon', 77.5946)
    
    feature_names = [
        'Population', 'Buildings', 'Roads', 'Intersections', 'Highway_Distance',
        'Builtup', 'Vegetation', 'Water', 'Night_Lights',
        'Schools', 'Hospitals', 'Police', 'Bus_Stops',
        'Elevation', 'Slope', 'NDVI', 'NDBI', 'Commercial'
    ]
    
    cell_size = 0.009
    transform = from_origin(
        lon - 0.144,
        lat + 0.144,
        cell_size,
        cell_size
    )
    
    for i, name in enumerate(feature_names[:10]):  # Limit to 10 key features
        tiff_path = os.path.join(DATA_ROOT, aoi_name, 'qgis', f'{aoi_name}_{name}.tif')
        
        with rasterio.open(
            tiff_path,
            'w',
            driver='GTiff',
            height=gis.shape[0],
            width=gis.shape[1],
            count=1,
            dtype=gis.dtype,
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(gis[:, :, i], 1)
        
        print(f"Created feature GeoTIFF: {tiff_path}")


def create_batch_file(aoi_name):
    """Create a batch file to open all layers in QGIS"""
    
    batch_content = f'''# Batch script to load layers in QGIS
# Run in QGIS Python Console

import os
import qgis.core

aoi = "{aoi_name}"
base_path = r"{os.path.join(DATA_ROOT, aoi_name, 'qgis')}"

# Load layers
layers = [
    ("Risk Map", os.path.join(base_path, f"{{aoi}}_risk.tif")),
    ("Human Exposure", os.path.join(base_path, f"{{aoi}}_Human_Exposure.tif")),
    ("Infrastructure", os.path.join(base_path, f"{{aoi}}_Infrastructure.tif")),
    ("Urban Activity", os.path.join(base_path, f"{{aoi}}_Urban_Activity.tif")),
    ("Environmental Buffer", os.path.join(base_path, f"{{aoi}}_Environmental_Buffer.tif")),
    ("Grid", os.path.join(base_path, f"{{aoi}}_grid.geojson")),
]

for name, path in layers:
    if os.path.exists(path):
        layer = qgis.core.QgsRasterLayer(path, name)
        if layer.isValid():
            QgsProject.instance().addMapLayer(layer)
        else:
            print(f"Failed to load: {{path}}")
'''
    
    batch_path = os.path.join(DATA_ROOT, aoi_name, 'qgis', 'load_layers.py')
    with open(batch_path, 'w') as f:
        f.write(batch_content)
    
    print(f"Created batch file: {batch_path}")


def process_all_aois():
    """Process all AOIs"""
    
    for aoi_name in AOIS.keys():
        print(f"\n{'='*60}")
        print(f"Processing: {aoi_name}")
        print('='*60)
        
        # Create QGIS project
        create_qgis_project_template(aoi_name)
        
        # Create grid
        create_geojson_grid(aoi_name)
        
        # Create risk map
        create_risk_geotiff(aoi_name)
        
        # Create components
        create_component_geotiffs(aoi_name)
        
        # Create feature GeoTIFFs
        create_feature_geotiffs(aoi_name)
        
        # Create batch file
        create_batch_file(aoi_name)


if __name__ == "__main__":
    process_all_aois()
    print("\nAll QGIS projects created!")
    print("   Open each .qgs file in QGIS to view the data.")
