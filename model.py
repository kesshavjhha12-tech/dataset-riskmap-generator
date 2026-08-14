"""
GCM-HAIRNet: Geographic Context Multimodal Human Activity Inference and Risk Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# GEOGRAPHIC POSITIONAL ENCODING
# ============================================================

class GeographicPositionalEncoding(nn.Module):
    """Learnable Geographic Positional Encoding"""
    def __init__(self, num_tokens, token_dim):
        super().__init__()
        self.pos_embeddings = nn.Parameter(torch.randn(num_tokens, token_dim))
    
    def forward(self, x):
        return x + self.pos_embeddings.unsqueeze(0)


# ============================================================
# GEOGRAPHIC BIAS MATRIX
# ============================================================

class GeographicBiasMatrix(nn.Module):
    """Learnable Geographic Bias Matrix for Self-Attention"""
    def __init__(self, num_tokens, sigma=3.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.sigma = sigma
        self.register_buffer('g0', self._create_gaussian_prior())
        self.g_learnable = nn.Parameter(torch.randn(num_tokens, num_tokens) * 0.01)
    
    def _create_gaussian_prior(self):
        grid_size = int(math.sqrt(self.num_tokens))
        coords = torch.arange(grid_size).float()
        x = coords.unsqueeze(0).repeat(grid_size, 1)
        y = coords.unsqueeze(1).repeat(1, grid_size)
        coords_flat = torch.stack([x.flatten(), y.flatten()], dim=1)
        diff = coords_flat.unsqueeze(1) - coords_flat.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        return torch.exp(-dist_sq / (2 * self.sigma ** 2))
    
    def forward(self):
        return self.g0 + self.g_learnable


# ============================================================
# MULTI-HEAD GEOGRAPHIC ATTENTION
# ============================================================

class MultiHeadGeographicAttention(nn.Module):
    """Multi-Head Self-Attention with Geographic Bias"""
    def __init__(self, token_dim, num_heads, num_tokens, dropout=0.1):
        super().__init__()
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        
        self.qkv = nn.Linear(token_dim, token_dim * 3)
        self.proj = nn.Linear(token_dim, token_dim)
        self.dropout = nn.Dropout(dropout)
        self.geographic_bias = GeographicBiasMatrix(num_tokens)
        
    def forward(self, x):
        batch_size, num_tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        g_bias = self.geographic_bias()
        attn = attn + g_bias.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, self.token_dim)
        out = self.proj(out)
        return out


# ============================================================
# GEOGRAPHIC CONTEXT TRANSFORMER BLOCK
# ============================================================

class GeographicContextTransformerBlock(nn.Module):
    """Single GCT Block with Geographic Attention"""
    def __init__(self, token_dim, num_heads, num_tokens, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(token_dim)
        self.attn = MultiHeadGeographicAttention(token_dim, num_heads, num_tokens, dropout)
        self.norm2 = nn.LayerNorm(token_dim)
        mlp_dim = int(token_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, token_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# GEOGRAPHIC CONTEXT TRANSFORMER
# ============================================================

class GeographicContextTransformer(nn.Module):
    """Geographic Context Transformer"""
    def __init__(self, token_dim, num_heads, num_tokens, num_layers=6, dropout=0.1):
        super().__init__()
        self.pos_encoding = GeographicPositionalEncoding(num_tokens, token_dim)
        self.blocks = nn.ModuleList([
            GeographicContextTransformerBlock(token_dim, num_heads, num_tokens, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(token_dim)
    
    def forward(self, x):
        x = self.pos_encoding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


# ============================================================
# GIS ENCODER
# ============================================================

class GISEncoder(nn.Module):
    """Lightweight CNN Encoder for GIS Features"""
    def __init__(self, in_channels=18, out_dim=512):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, out_dim, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(out_dim)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # x: (batch, 18, 32, 32)
        x = self.relu(self.bn1(self.conv1(x)))    # (batch, 64, 32, 32)
        x = self.pool1(x)                          # (batch, 64, 16, 16)
        
        x = self.relu(self.bn2(self.conv2(x)))    # (batch, 128, 16, 16)
        x = self.pool2(x)                          # (batch, 128, 8, 8)
        
        x = self.relu(self.bn3(self.conv3(x)))    # (batch, 256, 8, 8)
        x = self.relu(self.bn4(self.conv4(x)))    # (batch, 512, 8, 8)
        
        return x


# ============================================================
# VISUAL ENCODER (SwinV2)
# ============================================================

class VisualEncoder(nn.Module):
    """SwinV2-based Visual Encoder"""
    def __init__(self, out_dim=512):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm: pip install timm")
        
        self.backbone = timm.create_model('swinv2_tiny_window16_256', pretrained=True)
        self.backbone.reset_classifier(0)
        self.proj = nn.Conv2d(768, out_dim, 1)
        
    def forward(self, x):
        # x: (batch, 3, 256, 256)
        features = self.backbone.forward_features(x)  # (batch, 8, 8, 768)
        features = features.permute(0, 3, 1, 2)       # (batch, 768, 8, 8)
        features = self.proj(features)                # (batch, 512, 8, 8)
        return features


# ============================================================
# DECODER
# ============================================================

class Decoder(nn.Module):
    """Progressive Upsampling Decoder - 8x8 → 32x32"""
    def __init__(self, token_dim=512):
        super().__init__()
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(token_dim, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, 512, 8, 8)
        x = self.up1(x)   # (batch, 256, 16, 16)
        x = self.up2(x)   # (batch, 128, 32, 32)
        x = self.final(x) # (batch, 1, 32, 32)
        return x


# ============================================================
# GCM-HAIRNet MAIN MODEL
# ============================================================

class GCMHAIRNet(nn.Module):
    """Complete GCM-HAIRNet Model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Visual encoder - outputs (batch, 512, 8, 8)
        self.visual_encoder = VisualEncoder(out_dim=512)
        
        # GIS encoder - outputs (batch, 512, 8, 8)
        self.gis_encoder = GISEncoder(in_channels=18, out_dim=512)
        
        # Fusion
        self.fusion = nn.Conv2d(1024, 512, 1)
        
        # Transformer
        token_dim = 512
        num_tokens = 8 * 8  # 64 tokens
        
        self.transformer = GeographicContextTransformer(
            token_dim=token_dim,
            num_heads=8,
            num_tokens=num_tokens,
            num_layers=6,
            dropout=0.1
        )
        
        # Decoder
        self.decoder = Decoder(token_dim=token_dim)
    
    def forward(self, image, gis):
        """
        Args:
            image: (batch, 3, 256, 256) Sentinel-2 RGB
            gis: (batch, 18, 32, 32) GIS features
        
        Returns:
            risk: (batch, 1, 32, 32) Predicted risk map
        """
        # Encode visual features
        visual_features = self.visual_encoder(image)  # (batch, 512, 8, 8)
        
        # Encode GIS features
        gis_features = self.gis_encoder(gis)          # (batch, 512, 8, 8)
        
        # Fuse features
        fused = torch.cat([visual_features, gis_features], dim=1)  # (batch, 1024, 8, 8)
        fused = self.fusion(fused)                                 # (batch, 512, 8, 8)
        
        # Convert to tokens
        batch_size = fused.shape[0]
        tokens = fused.flatten(2).transpose(1, 2)  # (batch, 64, 512)
        
        # Apply Geographic Context Transformer
        tokens = self.transformer(tokens)          # (batch, 64, 512)
        
        # Convert back to spatial
        spatial = tokens.transpose(1, 2).reshape(batch_size, 512, 8, 8)
        
        # Decode to risk map
        risk = self.decoder(spatial)               # (batch, 1, 32, 32)
        
        return risk


# ============================================================
# LOSS FUNCTION
# ============================================================

class GCMHAIRNetLoss(nn.Module):
    """Multi-Component Loss for GCM-HAIRNet"""
    def __init__(self, mse_weight=0.45, mae_weight=0.25, ssim_weight=0.20, edge_weight=0.10):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        
    def compute_ssim(self, pred, target):
        """Simple SSIM approximation"""
        pred_mean = pred.mean()
        target_mean = target.mean()
        pred_var = pred.var()
        target_var = target.var()
        cov = ((pred - pred_mean) * (target - target_mean)).mean()
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = (2 * pred_mean * target_mean + c1) * (2 * cov + c2) / \
               (pred_mean**2 + target_mean**2 + c1) / \
               (pred_var + target_var + c2)
        
        return ssim.clamp(0, 1)
    
    def compute_edge_loss(self, pred, target):
        """Edge loss to preserve boundaries"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        sobel_x = sobel_x.view(1, 1, 3, 3).to(pred.device)
        sobel_y = sobel_y.view(1, 1, 3, 3).to(pred.device)
        
        pred_edges = torch.sqrt(
            F.conv2d(pred, sobel_x, padding=1)**2 + 
            F.conv2d(pred, sobel_y, padding=1)**2
        )
        target_edges = torch.sqrt(
            F.conv2d(target, sobel_x, padding=1)**2 + 
            F.conv2d(target, sobel_y, padding=1)**2
        )
        
        return F.l1_loss(pred_edges, target_edges)
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        ssim_loss = 1 - self.compute_ssim(pred, target)
        edge_loss = self.compute_edge_loss(pred, target)
        
        total_loss = (
            self.mse_weight * mse_loss +
            self.mae_weight * mae_loss +
            self.ssim_weight * ssim_loss +
            self.edge_weight * edge_loss
        )
        
        return total_loss, {
            'mse': mse_loss.item(),
            'mae': mae_loss.item(),
            'ssim': ssim_loss.item(),
            'edge': edge_loss.item()
        }