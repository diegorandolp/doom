import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from sample_factory.cfg.arguments import parse_sf_args
from sample_factory.train import run_rl
from sample_factory.model.encoder import Encoder
from sample_factory.model.decoder import Decoder
from sample_factory.model.core import ModelCore
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.utils.typing import Config, ObsSpace, ActionSpace

# --- 1. SOTA Encoder: ResNet-Impala with GroupNorm ---
class ImpalaResidualBlock(nn.Module):
    """
    The building block of the SOTA Impala architecture.
    Uses GroupNorm (stable for RL) instead of BatchNorm.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, channels) 
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        inputs = x
        out = F.relu(x)
        out = self.conv1(out)
        out = self.gn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        return out + inputs  # Residual connection

class CustomEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)
        
        # Detect input shape (Handle both dict and direct tensor spaces)
        if hasattr(obs_space, "keys"):
            shape = obs_space["obs"].shape 
        else:
            shape = obs_space.shape
            
        input_ch = shape[0] 

        # SOTA Config: Deeper stacks [64, 128, 128] for high visual acuity
        self.depths = [32, 64, 64] 
        
        layers = []
        current_channels = input_ch
        
        for i, depth in enumerate(self.depths):
            # Impala Stack: Conv -> MaxPool -> ResBlock -> ResBlock
            layers.append(nn.Conv2d(current_channels, depth, kernel_size=3, padding=1))
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            layers.append(ImpalaResidualBlock(depth))
            layers.append(ImpalaResidualBlock(depth))
            current_channels = depth
            
        self.features = nn.Sequential(*layers)
        
        # Calculate output size dynamically
        with torch.no_grad():
            # Assume standard resolution if unknown, or use actual
            h, w = shape[1], shape[2]
            dummy_input = torch.zeros(1, input_ch, h, w)
            out = self.features(dummy_input)
            self.encoder_out_size = out.numel()

        # Compression layer to feed into Core
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(self.encoder_out_size, 512), 
            nn.ReLU()
        )
        self.final_out_size = 512

    def forward(self, obs_dict):
        # Sample Factory always passes a dictionary
        x = obs_dict['obs'] 
        x = self.features(x)
        x = self.fc(x)
        return x

    def get_out_size(self):
        return self.final_out_size



def register_model_components():
    # Register the full ActorCritic to ensure our forward pass logic (Dueling) is used
    global_model_factory().register_encoder_factory(CustomEncoder)
