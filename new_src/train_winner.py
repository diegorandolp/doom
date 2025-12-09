import sys
import random
import numpy as np
import torch
import gymnasium as gym

# Import VizDoom to access Button enums for masking
try:
    import vizdoom.vizdoom as vizdoom
    from vizdoom import Button
except ImportError:
    import vizdoom
    from vizdoom import Button

# Sample Factory Imports
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

# VizDoom Specific Imports
from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
from sf_examples.vizdoom.doom.doom_utils import make_doom_env_from_spec, DoomSpec
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.doom.action_space import doom_action_space_discretized
from custom_model import register_custom_model

# --- CONFIGURATION: Mapping Scenarios to Config Files ---
SCENARIO_CONFIGS = {
    # Standard Scenarios
    "doom_basic": "basic.cfg",
    "doom_deadly_corridor": "deadly_corridor.cfg",
    "doom_defend_the_center": "defend_the_center.cfg",
    "doom_health_gathering_supreme": "health_gathering_supreme.cfg",
    "doom_my_way_home": "my_way_home.cfg",
    
    # New Scenarios (Added by request)
    "doom_deathmatch": "deathmatch.cfg", 
    "doom_health_gathering": "health_gathering.cfg",
    "doom_battle": "battle.cfg",
    "doom_battle2": "battle2_continuous_turning.cfg", # Assuming standard name or verify file exists
    "doom_defend_the_line": "defend_the_line.cfg",
    "doom_two_colors_hard": "two_colors_hard.cfg" # Added .cfg extension just in case
}

# --- REVISED ACTION MASKING RULES ---

# We define the structure of doom_action_space_discretized()
# Total Logits = 24
# Head 0: Move F/B (3 options: No-op, Fwd, Back)      -> Indices 0, 1, 2
# Head 1: Strafe   (3 options: No-op, Right, Left)    -> Indices 3, 4, 5
# Head 2: Weapon   (3 options: No-op, Prev, Next)     -> Indices 6, 7, 8
# Head 3: Attack   (2 options: No-op, Attack)         -> Indices 9, 10
# Head 4: Sprint   (2 options: No-op, Sprint)         -> Indices 11, 12
# Head 5: Turn     (11 options: -10 to +10 delta)     -> Indices 13 to 23

class ScenarioRules:
    # 1. PACIFIST (Health Gathering, My Way Home)
    # Allow: Move, Strafe, Sprint, Turn
    # Ban: Weapon, Attack
    PACIFIST = {
        "move_fb": True,
        "strafe": True,
        "weapon": False,
        "attack": False,
        "sprint": True,
        "turn": True
    }

    # 2. TURRET (Defend the Center/Line)
    # Allow: Turn, Attack, Weapon (maybe)
    # Ban: Move F/B, Strafe (Agent cannot move in these maps)
    TURRET = {
        "move_fb": False,
        "strafe": False,
        "weapon": True,
        "attack": True,
        "sprint": False,
        "turn": True
    }

    # 3. FULL COMBAT (Deathmatch, Deadly Corridor, Battle)
    # Allow: Everything
    COMBAT = {
        "move_fb": True,
        "strafe": True,
        "weapon": True,
        "attack": True,
        "sprint": True,
        "turn": True
    }

# Map Scenarios to Rules
SCENARIO_MASK_CONFIG = {
    "doom_health_gathering": ScenarioRules.PACIFIST,
    "doom_health_gathering_supreme": ScenarioRules.PACIFIST,
    "doom_my_way_home": ScenarioRules.PACIFIST,
    "doom_two_colors_hard": ScenarioRules.PACIFIST,
    
    "doom_defend_the_center": ScenarioRules.TURRET,
    "doom_defend_the_line": ScenarioRules.TURRET,
    
    "doom_basic": ScenarioRules.COMBAT,
    "doom_deadly_corridor": ScenarioRules.COMBAT,
    "doom_deathmatch": ScenarioRules.COMBAT,
    "doom_battle": ScenarioRules.COMBAT,
    "doom_battle2": ScenarioRules.COMBAT
}

class ActionMaskWrapper(gym.Wrapper):
    def __init__(self, env, scenario_name):
        super().__init__(env)
        
        # 1. Get Rules
        rules = SCENARIO_MASK_CONFIG.get(scenario_name, ScenarioRules.COMBAT)
        
        # 2. Build the Static Mask (Size 24 for discretized space)
        # We start with ALL ones (everything allowed)
        mask = np.ones(24, dtype=np.float32)
        
        # Head 0: Move F/B [No-op, Fwd, Back] (Indices 0-2)
        if not rules["move_fb"]:
            mask[1] = 0.0 # Ban Forward
            mask[2] = 0.0 # Ban Backward
            
        # Head 1: Strafe [No-op, Right, Left] (Indices 3-5)
        if not rules["strafe"]:
            mask[4] = 0.0
            mask[5] = 0.0
            
        # Head 2: Weapon [No-op, Prev, Next] (Indices 6-8)
        if not rules["weapon"]:
            mask[7] = 0.0
            mask[8] = 0.0
            
        # Head 3: Attack [No-op, Fire] (Indices 9-10)
        if not rules["attack"]:
            mask[10] = 0.0
            
        # Head 4: Sprint [No-op, Sprint] (Indices 11-12)
        if not rules["sprint"]:
            mask[12] = 0.0
            
        # Head 5: Turn (Indices 13-23) - Always allowed unless specified
        if not rules["turn"]:
            mask[13:] = 0.0
            mask[13 + 5] = 1.0 # Allow the middle index (No-op turn) so it doesn't crash
            
        self.static_mask = mask
        
        # 3. Update Observation Space
        current_obs_space = self.env.observation_space
        if isinstance(current_obs_space, gym.spaces.Dict):
            spaces = current_obs_space.spaces.copy()
        else:
            spaces = {"obs": current_obs_space}
            
        spaces["action_mask"] = gym.spaces.Box(
            low=0, high=1, shape=(24,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(spaces)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs["action_mask"] = self.static_mask
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs["action_mask"] = self.static_mask
        return obs, reward, terminated, truncated, info

def make_multi_task_doom_env(full_env_name, cfg=None, env_config=None, render_mode=None, **kwargs):
    """
    Creates a Doom environment with a random scenario and applies Action Masking.
    """
    # 1. Pick a random scenario name
    scenario_name = random.choice(list(SCENARIO_CONFIGS.keys()))
    config_file = SCENARIO_CONFIGS[scenario_name]
    
    # 2. Create the Spec
    # Use discretized space to cover all possible moves across all games
    env_spec = DoomSpec(scenario_name, config_file, doom_action_space_discretized())
    
    # 3. Create the Base Environment
    env = make_doom_env_from_spec(
        env_spec, 
        full_env_name,
        cfg=cfg, 
        env_config=env_config, 
        render_mode=render_mode, 
        **kwargs
    )
    
    # 4. Wrap with Action Masking
    # This ensures the agent sees the mask specific to the chosen scenario
    env = ActionMaskWrapper(env, scenario_name)
    
    return env

def register_custom_components():
    """Register the custom environment AND the custom model encoder."""
    register_env("doom_multi_task", make_multi_task_doom_env)
    
    # Register your NEW Architecture (Transformer + ConvStem)
    register_custom_model()
    # Note: We don't register make_vizdoom_encoder anymore because register_custom_model 
    # overrides the encoder factory with your ViT class.

def parse_vizdoom_cfg(argv=None, evaluation=False):
    parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg

def verify_gpu_or_die():
    print("=" * 50)
    print("HARDWARE VERIFICATION")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("❌ CRITICAL ERROR: No GPU detected via torch.cuda.is_available()!")
        sys.exit(1)
    
    try:
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"✅ GPU DETECTED: {gpu_name}")
    except Exception as e:
        print(f"⚠️ GPU Detection Warning: {e}")

    print("=" * 50)

def main():
    verify_gpu_or_die()
    register_custom_components()
    cfg = parse_vizdoom_cfg()
    status = run_rl(cfg)
    return status

if __name__ == "__main__":
    sys.exit(main())
