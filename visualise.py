"""
Visualization Script - Works with any image size
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.colors import LinearSegmentedColormap


def load_visualization_data():
    """Load saved predictions and data"""
    save_dir = './visualization_data'
    
    if not os.path.exists(save_dir):
        print("❌ No visualization data found. Run save_predictions.py first.")
        return None
    
    images = np.load(os.path.join(save_dir, 'images.npy'))    # (N, 1, 3, 256, 256)
    preds = np.load(os.path.join(save_dir, 'preds.npy'))      # (N, 1, 32, 32)
    targets = np.load(os.path.join(save_dir, 'targets.npy'))  # (N, 1, 32, 32)
    
    with open(os.path.join(save_dir, 'aois.json'), 'r') as f:
        aois = json.load(f)
    
    print(f"Loaded {len(images)} samples")
    print(f"  Images shape: {images.shape}")
    print(f"  Predictions shape: {preds.shape}")
    print(f"  Targets shape: {targets.shape}")
    
    return images, preds, targets, aois


def visualize_results(num_samples=4):
    """Create publication-quality visualization"""
    
    data = load_visualization_data()
    if data is None:
        return
    
    images, preds, targets, aois = data
    
    # Select samples
    num_samples = min(num_samples, len(images))
    indices = np.linspace(0, len(images)-1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    # Color map for risk
    cmap = LinearSegmentedColormap.from_list('risk', ['#2c7bb6', '#ffffbf', '#d7191c'])
    
    for row, idx in enumerate(indices):
        # Get image: (1, 3, 256, 256) -> (256, 256, 3)
        img = images[idx]
        if img.ndim == 4:
            img = img.squeeze(0)  # (3, 256, 256)
        img = img.transpose(1, 2, 0)  # (256, 256, 3)
        img = np.clip(img, 0, 1)
        
        # Ground Truth and Prediction
        gt = targets[idx].squeeze()  # (32, 32)
        pred = preds[idx].squeeze()  # (32, 32)
        error = np.abs(gt - pred)
        
        aoi_name = aois[idx]
        
        # Plot satellite image
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f'AOI: {aoi_name}')
        axes[row, 0].axis('off')
        
        # Ground Truth
        im1 = axes[row, 1].imshow(gt, cmap=cmap, vmin=0, vmax=1)
        axes[row, 1].set_title('Ground Truth')
        axes[row, 1].axis('off')
        
        # Prediction
        im2 = axes[row, 2].imshow(pred, cmap=cmap, vmin=0, vmax=1)
        axes[row, 2].set_title('Prediction')
        axes[row, 2].axis('off')
        
        # Error
        im3 = axes[row, 3].imshow(error, cmap='Reds', vmin=0, vmax=0.2)
        axes[row, 3].set_title(f'Error (MAE: {error.mean():.3f})')
        axes[row, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('visualization_results.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: visualization_results.png")
    plt.show()


def plot_training_history():
    """Plot training and validation loss"""
    history_path = 'training_history.json'
    if os.path.exists(history_path):
        import json
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: training_history.png")
        plt.show()
    else:
        print("⚠️ No training_history.json found")


if __name__ == "__main__":
    print("=" * 60)
    print("GCM-HAIRNet Visualization")
    print("=" * 60)
    
    visualize_results(num_samples=4)
    plot_training_history()