# Urban Risk Dataset & Risk Map Generator

A complete, reproducible pipeline for generating UAV urban risk assessment datasets from satellite imagery and GIS data.

This pipeline downloads Sentinel-2 satellite imagery and 18 GIS features for any location, builds interpretable risk components, and generates ground truth risk maps using a Tiny CNN.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Dataset Specification](#dataset-specification)
5. [GIS Features](#gis-features)
6. [Installation](#installation)
7. [Quick Start](#quick-start)
8. [Pipeline Steps](#pipeline-steps)
9. [Risk Map Generation](#risk-map-generation)
10. [Output Format](#output-format)
11. [QGIS Integration](#qgis-integration)
12. [Citation](#citation)
13. [License](#license)

---

## Overview

Urban UAV navigation is challenging due to dense infrastructure, high population concentrations, and complex urban morphology. Existing risk assessment approaches rely on manually designed formulas that are static and difficult to generalize.

This pipeline provides a data-driven solution by:

1. Collecting satellite imagery and GIS data for any urban area
2. Extracting 18 GIS features at 1km resolution
3. Building four interpretable risk components
4. Generating ground truth risk maps using a Tiny CNN
5. Producing a complete, normalized dataset ready for deep learning training

The generated dataset is designed to train GCM-HAIRNet, a Geographic Context Transformer for UAV risk assessment.

---

## Key Features

- 60+ Indian cities supported (expandable to any location)
- 18 GIS features from public datasets
- Sentinel-2 satellite imagery at 10m resolution
- Interpretable risk components (HE, IC, UA, EB)
- Tiny CNN (4->32->16->1) for risk map generation
- Automatic data collection and processing
- QGIS integration for visualization
- Fully reproducible with open data

---

## Project Structure

```
gcm_hairnet/
│
├── data/                                 # All dataset files
│   ├── processed/                        # Final normalized dataset
│   │   ├── images/                       # Satellite images (256x256x3)
│   │   ├── gis/                          # GIS features (32x32x18)
│   │   ├── labels/                       # Risk maps (32x32)
│   │   ├── metadata/                     # AOI metadata
│   │   └── splits.json                   # Train/val/test splits
│   │
│   └── [AOI_NAME]/                       # Raw data per AOI
│       ├── raw/                          # Downloaded data
│       ├── gis/                          # Extracted GIS features
│       ├── labels/                       # Generated risk maps
│       └── [AOI]_features.csv            # Features in CSV format
│
├── checkpoints/                          # Model checkpoints
│   └── best_model.pth                    # Best trained model
│
├── config.py                             # Configuration (AOIs, features, paths)
├── aoi_list.py                           # AOI management
├── data_collector.py                     # Download Sentinel-2 + GIS data
├── feature_extractor.py                  # Extract 18 GIS features
├── component_builder.py                  # Build HE, IC, UA, EB components
├── risk_generator.py                     # Generate risk maps (Tiny CNN)
├── dataset_builder.py                    # Assemble final dataset
├── normalise.py                          # Normalize dataset
├── model.py                              # GCM-HAIRNet model (optional)
├── train.py                              # Training script (optional)
├── evaluate.py                           # Evaluation script (optional)
├── visualize.py                          # Visualization script (optional)
├── save_predictions.py                   # Save predictions for visualization
├── run_pipeline.py                       # Master pipeline script
├── requirements.txt                      # Dependencies
└── README.md                             # This file
```

---

## Dataset Specification

### AOI Categories

| Category | Number of AOIs | Examples |
|----------|----------------|----------|
| Metropolitan | 15 | Bengaluru, Mumbai, Delhi, Chennai, Hyderabad |
| Industrial | 4 | Peenya, Electronic City, Hosur, Chennai Port |
| Semi-Urban | 4 | Mysuru, Bhubaneswar, Mangalore, Hoskote |
| Rural | 4 | Bannerghatta, Ramanagara, Mandya, Chikkaballapur |
| Additional | 4+ | Expandable to 60+ cities |

### AOI Specification

- AOI Size: 32 km x 32 km
- Grid: 32 x 32 cells
- Cell Size: 1 km x 1 km
- Total Cells per AOI: 1,024
- Total Samples (60 AOIs): 61,440

---

## GIS Features

18 GIS features are extracted for each 1km cell:

| Feature | Source | Description |
|---------|--------|-------------|
| Population Density | WorldPop | People per km2 |
| Building Density | OSM | Building coverage |
| Road Density | OSM | Road length per km2 |
| Road Intersection Density | OSM | Intersections per km2 |
| Distance to Highway | OSM | Distance to nearest highway |
| Built-up Percentage | ESA WorldCover | Percentage of built-up area |
| Vegetation Percentage | ESA WorldCover | Percentage of vegetation |
| Water Percentage | ESA WorldCover | Percentage of water |
| Night Lights | VIIRS | Light intensity |
| School Count | OSM | Number of schools |
| Hospital Count | OSM | Number of hospitals |
| Police Count | OSM | Number of police stations |
| Bus Stop Count | OSM | Number of bus stops |
| Elevation | Copernicus DEM | Meters above sea level |
| Slope | Copernicus DEM | Slope in degrees |
| NDVI | Sentinel-2 | Vegetation health index |
| NDBI | Sentinel-2 | Built-up index |
| Commercial Percentage | OSM | Percentage of commercial area |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/gcm_hairnet.git
cd gcm_hairnet
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# or
.venv\Scripts\activate         # Windows

# Using conda
conda create -n gcm_env python=3.10
conda activate gcm_env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch (Optional - for model training)

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Quick Start

Run the complete pipeline:

```bash
python run_pipeline.py
```

This will:
1. Collect data for all AOIs
2. Extract GIS features
3. Build risk components
4. Generate risk maps
5. Assemble the final dataset

---

## Pipeline Steps

### Step 1: Data Collection

```bash
python data_collector.py
```

Downloads:
- Sentinel-2 satellite imagery (10m resolution)
- WorldPop population data
- OSM buildings, roads, POIs
- ESA WorldCover land cover
- VIIRS night lights
- Copernicus DEM elevation data

### Step 2: Feature Extraction

```bash
python feature_extractor.py
```

Extracts 18 GIS features per 1km cell and saves as a (32x32x18) tensor.

### Step 3: Component Building

```bash
python component_builder.py
```

Builds 4 interpretable components from the 18 GIS features:
- Human Exposure (HE)
- Infrastructure Complexity (IC)
- Urban Activity (UA)
- Environmental Buffer (EB)

### Step 4: Risk Map Generation

```bash
python risk_generator.py
```

Generates ground truth risk maps using Tiny CNN architecture (4->32->16->1).

### Step 5: Dataset Assembly

```bash
python dataset_builder.py
```

Assembles the final dataset with train/val/test splits.

### Step 6: Normalization

```bash
python normalise.py
```

Normalizes all features to the range [0, 1].

---

## Risk Map Generation

### Interpretable Risk Components

| Component | Full Form | Features Included |
|-----------|-----------|-------------------|
| HE | Human Exposure | Population, Buildings, Schools, Hospitals, Police, Bus Stops |
| IC | Infrastructure Complexity | Roads, Intersections, Highway Distance |
| UA | Urban Activity | Night Lights, Built-up, Commercial |
| EB | Environmental Buffer | Vegetation, Water, NDVI, Slope, Elevation |

### Tiny CNN Architecture

```
Input: (4, 32, 32) -> [HE, IC, UA, EB]
       |
       v
Conv3x3 (4 -> 32) + ReLU
       |
       v
Conv3x3 (32 -> 16) + ReLU
       |
       v
Conv1x1 (16 -> 1) + Sigmoid
       |
       v
Output: (1, 32, 32) -> Risk Map (0 to 1)
```

Total Parameters: ~160

### Why a Tiny CNN?

- Fixed equations assume known relationships between risk factors
- Tiny CNN learns the mapping from data
- Convolutions capture local spatial interactions
- Lightweight and interpretable
- Prevents overfitting

---

## Output Format

### Per AOI

```
data/processed/
├── images/
│   └── [AOI]_image.npy        (256x256x3)  - Satellite image
├── gis/
│   └── [AOI]_gis.npy          (32x32x18)   - GIS features
└── labels/
    └── [AOI]_risk.npy         (32x32)      - Risk map (ground truth)
```

### CSV Export

```
data/[AOI]/[AOI]_features.csv
```

Contains 1,024 rows (32x32 cells) with:
- cell_id, row, col, latitude, longitude
- All 18 GIS features
- Risk score (if generated)

### QGIS Export

```
data/[AOI]/qgis/
├── [AOI].qgs                  - QGIS project file
├── [AOI]_grid.geojson         - 32x32 grid
├── [AOI]_risk.tif             - Risk map GeoTIFF
├── [AOI]_HE.tif               - Human Exposure GeoTIFF
├── [AOI]_IC.tif               - Infrastructure GeoTIFF
├── [AOI]_UA.tif               - Urban Activity GeoTIFF
└── [AOI]_EB.tif               - Environmental Buffer GeoTIFF
```

---

## QGIS Integration

### Open in QGIS

```bash
qgis data/[AOI]/qgis/[AOI].qgs
```

### Load Layers Manually

1. Open QGIS
2. Layer -> Add Layer -> Add Raster Layer
3. Select `[AOI]_risk.tif`
4. Layer -> Add Layer -> Add Vector Layer
5. Select `[AOI]_grid.geojson`

### Styling Recommendations

| Layer | Color Ramp | Description |
|-------|------------|-------------|
| Risk Map | RdYlGn (Red-Yellow-Green) | High risk = Red, Low risk = Green |
| HE | Oranges | High = More people/buildings |
| IC | Blues | High = Dense roads/intersections |
| UA | Purples | High = Night lights/commercial |
| EB | Greens | High = More vegetation/water |


---



## Acknowledgments

- Sentinel-2 for satellite imagery
- WorldPop for population data
- OpenStreetMap for infrastructure data
- ESA WorldCover for land cover data
- VIIRS for night lights data
- Copernicus DEM for elevation data

---

## Contact

For questions or collaborations: kesshavjhha12@gmail.com

---

## Keywords

Urban Risk Assessment, UAV Navigation, Dataset Generation, GIS, Remote Sensing, Deep Learning, Sentinel-2, Risk Mapping, Indian Cities, Geospatial Analysis
