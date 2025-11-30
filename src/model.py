import torch
import torch.nn as nn
import torch.nn.functional as F

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = self._res_block(out_channels)
        self.res2 = self._res_block(out_channels)

    def _res_block(self, channels):
        return nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.res1(x) + x
        x = self.res2(x) + x
        return x

class DoomAgent(nn.Module):
    def __init__(self, action_space_size, num_scenarios=13):
        super().__init__()
        
        # --- 1. VISUAL ENCODER (IMPALA) ---
        # Input: (Batch, 3, H, W) - We assume 3 RGB channels. 
        # If frame stacking is done in channel dim, adjust '3' to '12' (3*4 frames).
        # We will use 3 channels here and handle time via the GRU.
        self.encoder = nn.Sequential(
            ImpalaBlock(3, 32),   # Output: 32 x 64 x 36
            ImpalaBlock(32, 64),  # Output: 64 x 32 x 18
            ImpalaBlock(64, 64),  # Output: 64 x 16 x 9
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Latent size calculation: 64 * 16 * 9 = 9216
        self.fc_visual = nn.Linear(9216, 512)
        
        # --- 2. TEMPORAL MEMORY (GRU) ---
        # This replaces the need for Frame Stacking (mostly) and gives long-term memory
        self.gru = nn.GRUCell(input_size=512, hidden_size=512)
        
        # --- 3. TASK CONDITIONING ---
        # The agent needs to know WHICH scenario it is playing.
        self.task_embedding = nn.Embedding(num_scenarios, 64)
        
        # --- 4. HEADS ---
        # The Core Features = GRU State (512) + Task Info (64)
        core_size = 512 + 64
        
        # Actor: Decides what to do
        self.actor = nn.Linear(core_size, action_space_size)
        
        # Critic: Estimates value (PopArt style - Multi-Head)
        # Instead of complex normalization layers, we output ONE value per scenario.
        # This allows the network to predict "2000" for Survival and "100" for Puzzle independently.
        self.critic_heads = nn.Linear(core_size, num_scenarios)

    def forward(self, x, hidden_state, scenario_ids):
        """
        x: [Batch, 3, 128, 72] image
        hidden_state: [Batch, 512] previous memory
        scenario_ids: [Batch] integer ids of the map (0-12)
        """
        # 1. Vision
        # Pixel normalization (0-255 -> 0-1) should happen before this or inside
        x = x / 255.0 
        features = self.encoder(x)
        features = self.fc_visual(features)
        
        # 2. Memory
        new_hidden = self.gru(features, hidden_state)
        
        # 3. Task Context
        task_embed = self.task_embedding(scenario_ids)
        core_features = torch.cat([new_hidden, task_embed], dim=1)
        
        # 4. Outputs
        action_logits = self.actor(core_features)
        all_values = self.critic_heads(core_features)
        
        # Select the specific value head for the current scenario
        # Gather logic: we want all_values[batch_idx, scenario_ids[batch_idx]]
        curr_values = all_values.gather(1, scenario_ids.unsqueeze(1)).squeeze(1)
        
        return action_logits, curr_values, new_hidden
