import torch
import torch.nn as nn
import math

from sample_factory.model.encoder import Encoder
from sample_factory.model.core import ModelCore
from sample_factory.model.decoder import Decoder
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.utils.typing import Config, ObsSpace

# --- 1. The Conv Stem (Downsamples image to patches) ---
class ConvStem(nn.Module):
    def __init__(self, input_channels, feature_dim=64):
        super().__init__()
        # Input: [Batch, C, 128, 72] (Standard VizDoom size)
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0), # -> [32, 31, 17]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),  # -> [64, 14, 7]
            nn.ReLU(),
            nn.Conv2d(64, feature_dim, kernel_size=3, stride=1, padding=1), # -> [64, 14, 7]
            nn.ReLU(),
        )
        self.out_channels = feature_dim
    
    def forward(self, x):
        return self.main(x)

# --- 2. The Vision Transformer Encoder ---
class ViTEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)
        
        # Dimensions
        input_ch = obs_space['obs'].shape[0]
        self.conv_dim = 128  # Dimension of the feature map channels
        self.embed_dim = 256 # Dimension of the Transformer tokens
        self.num_heads = 4
        self.num_layers = 3  # Keep shallow for RL stability
        
        # 1. Conv Stem
        self.stem = ConvStem(input_ch, self.conv_dim)
        
        # Calculate spatial size after stem (assuming 128x72 input)
        # You might need to adjust strictly if resolution changes, 
        # but this works for standard 128x72
        self.grid_h = 14
        self.grid_w = 7
        self.num_patches = self.grid_h * self.grid_w
        
        # 2. Project Conv Features to Embeddings
        self.patch_embed = nn.Conv2d(self.conv_dim, self.embed_dim, kernel_size=1)
        
        # 3. Positional Encoding (Learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=self.num_heads, 
            dim_feedforward=512, 
            dropout=0.0, 
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # 5. Output Projector (Pooling to vector)
        self.ln = nn.LayerNorm(self.embed_dim)
        self.encoder_out_size = self.embed_dim

    def forward(self, obs_dict):
        x = obs_dict['obs'] # [Batch, C, H, W]
        
        # A. Pass through Conv Stem
        x = self.stem(x)    # [Batch, 128, 14, 7]
        
        # B. Project to Embeddings & Flatten
        x = self.patch_embed(x) # [Batch, 256, 14, 7]
        x = x.flatten(2)        # [Batch, 256, 98]
        x = x.transpose(1, 2)   # [Batch, 98, 256] -> [Batch, Seq_Len, Embed_Dim]
        
        # C. Add Positional Encoding
        x = x + self.pos_embed
        
        # D. Transformer Layers
        x = self.transformer(x) # [Batch, 98, 256]
        
        # E. Pooling (We use Mean Pooling over all patches)
        # This aggregates "Global Attention" into one vector
        x = self.ln(x)
        x = x.mean(dim=1)       # [Batch, 256]
        
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size

# --- 3. Registration Function ---
def register_custom_model():
    # We only need to register the Encoder. 
    # Sample Factory will automatically attach the standard RNN (Core) 
    # and Decoder (Heads) if we don't override them.
    # This is safer and cleaner than overriding the whole ActorCritic.
    global_model_factory().register_encoder_factory(ViTEncoder)
