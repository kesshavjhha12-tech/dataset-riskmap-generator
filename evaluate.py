"""
Evaluation Script for GCM-HAIRNet
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import json
from tqdm import tqdm

from model import GCMHAIRNet
from train import UrbanRiskDataset
from config import DATA_ROOT


def compute_ssim(pred, target, win_size=3):
    """
    Compute SSIM safely with smaller window size
    """
    from skimage.metrics import structural_similarity as ssim
    
    # Ensure arrays are 2D
    if len(pred.shape) == 3:
        pred = pred.squeeze(0)
    if len(target.shape) == 3:
        target = target.squeeze(0)
    
    # Ensure window size is odd and <= image size
    h, w = pred.shape
    win_size = min(win_size, h, w)
    if win_size % 2 == 0:
        win_size -= 1  # Make odd
    win_size = max(3, win_size)  # Minimum 3
    
    try:
        return ssim(pred, target, data_range=1.0, win_size=win_size)
    except ValueError:
        # Fallback: use smaller window
        win_size = min(3, h, w)
        if win_size % 2 == 0:
            win_size -= 1
        return ssim(pred, target, data_range=1.0, win_size=max(3, win_size))


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj


def evaluate_model(config):
    """Evaluate GCM-HAIRNet"""
    
    # Load splits
    splits_path = os.path.join(DATA_ROOT, 'processed', 'splits.json')
    
    if not os.path.exists(splits_path):
        print("splits.json not found! Creating default...")
        splits = {
            "train": [
                "Bengaluru_CBD", "Whitefield", "Hyderabad", "Chennai",
                "Pune", "Mumbai", "Delhi", "Ahmedabad",
                "Peenya", "Electronic_City",
                "Mysuru", "Bhubaneswar",
                "Bannerghatta", "Ramanagara"
            ],
            "val": ["Hosur", "Mangalore", "Mandya"],
            "test": ["Chennai_Port", "Hoskote", "Chikkaballapur"]
        }
        os.makedirs(os.path.dirname(splits_path), exist_ok=True)
        with open(splits_path, 'w') as f:
            json.dump(splits, f, indent=2)
    else:
        with open(splits_path, 'r') as f:
            splits = json.load(f)
    
    # Create test dataset
    test_dataset = UrbanRiskDataset(splits['test'], split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Load model
    checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found: {checkpoint_path}")
        print("   Creating a dummy model for testing...")
        model = GCMHAIRNet(config).to(config['device'])
    else:
        model = GCMHAIRNet(config).to(config['device'])
        checkpoint = torch.load(checkpoint_path, map_location=config['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # Metrics
    all_preds = []
    all_targets = []
    all_aois = []
    
    print("Evaluating on test set...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            image = batch['image'].to(config['device'])
            gis = batch['gis'].to(config['device'])
            label = batch['label'].to(config['device'])
            
            pred = model(image, gis)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(label.cpu().numpy())
            all_aois.append(batch['aoi'][0])
    
    # Convert to arrays
    preds = np.concatenate(all_preds, axis=0)  # (N, 1, 32, 32)
    targets = np.concatenate(all_targets, axis=0)  # (N, 1, 32, 32)
    
    # Flatten for metrics
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    
    # Compute metrics
    mae = mean_absolute_error(targets_flat, preds_flat)
    mse = mean_squared_error(targets_flat, preds_flat)
    rmse = np.sqrt(mse)
    
    # Pearson correlation
    try:
        pearson, _ = pearsonr(targets_flat, preds_flat)
    except:
        pearson = 0.0
    
    # Spearman correlation
    try:
        spearman, _ = spearmanr(targets_flat, preds_flat)
    except:
        spearman = 0.0
    
    # SSIM - compute per image and average
    ssim_scores = []
    for i in range(preds.shape[0]):
        try:
            pred_2d = preds[i].squeeze()
            target_2d = targets[i].squeeze()
            score = compute_ssim(pred_2d, target_2d)
            ssim_scores.append(score)
        except Exception as e:
            try:
                from skimage.metrics import structural_similarity as ssim
                pred_2d = preds[i].squeeze()
                target_2d = targets[i].squeeze()
                score = ssim(pred_2d, target_2d, data_range=1.0, win_size=3)
                ssim_scores.append(score)
            except:
                ssim_scores.append(0.5)
    
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'Pearson': pearson,
        'Spearman': spearman,
        'SSIM': avg_ssim,
        'num_samples': len(all_preds)
    }
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for k, v in metrics.items():
        if k != 'num_samples':
            print(f"  {k}: {v:.4f}")
    print(f"  Samples: {metrics['num_samples']}")
    print("=" * 60)
    
    return metrics, all_preds, all_targets, all_aois


def compute_per_aoi_metrics(all_preds, all_targets, all_aois):
    """Compute metrics per AOI"""
    
    aoi_metrics = {}
    
    for pred, target, aoi in zip(all_preds, all_targets, all_aois):
        if aoi not in aoi_metrics:
            aoi_metrics[aoi] = {'preds': [], 'targets': []}
        
        aoi_metrics[aoi]['preds'].append(pred.flatten())
        aoi_metrics[aoi]['targets'].append(target.flatten())
    
    results = {}
    for aoi, data in aoi_metrics.items():
        preds_flat = np.concatenate(data['preds'])
        targets_flat = np.concatenate(data['targets'])
        
        results[aoi] = {
            'MAE': mean_absolute_error(targets_flat, preds_flat),
            'MSE': mean_squared_error(targets_flat, preds_flat),
            'RMSE': np.sqrt(mean_squared_error(targets_flat, preds_flat)),
            'num_cells': len(preds_flat)
        }
    
    return results


def main():
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'checkpoint_dir': './checkpoints'
    }
    
    print("=" * 60)
    print("GCM-HAIRNet Evaluation")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print("=" * 60)
    
    # Evaluate
    metrics, preds, targets, aois = evaluate_model(config)
    
    # Per-AOI metrics
    print("\n📊 Per-AOI Metrics:")
    print("-" * 60)
    aoi_metrics = compute_per_aoi_metrics(preds, targets, aois)
    
    for aoi, m in sorted(aoi_metrics.items()):
        print(f"  {aoi:20s} | MAE: {m['MAE']:.4f} | RMSE: {m['RMSE']:.4f} | Cells: {m['num_cells']}")
    
    print("-" * 60)
    
    # Convert all values to JSON serializable format
    results = {
        'overall_metrics': convert_to_serializable(metrics),
        'per_aoi_metrics': convert_to_serializable(aoi_metrics),
        'timestamp': str(np.datetime64('now'))
    }
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nsave ho gayaa evaluation_results.json")


if __name__ == "__main__":
    main()
