"""
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
    
    # Step 1: Load AOIs
    print("\n📌 Step 1: Loading AOIs...")
    aoi_list = get_all_aois()
    print(f"   Total AOIs: {len(aoi_list)}")
    
    # Step 2: Collect Data
    print("\n📌 Step 2: Collecting Data...")
    collect_all_aois(aoi_list)
    
    # Step 3: Extract Features
    print("\n📌 Step 3: Extracting Features...")
    extract_all_aois(aoi_list)
    
    # Step 4: Build Components
    print("\n📌 Step 4: Building Components...")
    build_all_components(aoi_list)
    
    # Step 5: Train Tiny CNN
    print("\n📌 Step 5: Training Tiny CNN...")
    trainer = TinyCNNTrainer()
    X, y = trainer.prepare_training_data(aoi_list)
    model = trainer.train(X, y)
    trainer.save_model(os.path.join(DATA_ROOT, 'tiny_cnn.pth'))
    
    # Step 6: Generate Risk Maps
    print("\n📌 Step 6: Generating Risk Maps...")
    risk_generator = RiskGenerator(os.path.join(DATA_ROOT, 'tiny_cnn.pth'))
    for aoi in aoi_list:
        risk_generator.generate_for_aoi(aoi.name)
    
    # Step 7: Build Dataset
    print("\n📌 Step 7: Building Final Dataset...")
    dataset_builder = DatasetBuilder()
    dataset_builder.build(aoi_list)
    
    print("\n" + "=" * 60)
    print("✅ Pipeline Complete!")
    print(f"   Dataset saved to: {DATA_ROOT}/processed")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()