"""
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