import gymnasium as gym
import numpy as np
import os

from sample_factory.envs.env_utils import register_env
from sf_examples.vizdoom.doom.doom_utils import make_doom_env_from_spec, DoomSpec
from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
from sf_examples.vizdoom.doom.action_space import doom_action_space_discretized
from sample_factory.algo.utils.context import global_model_factory
from vizdoom import Button

# --- 1. DEFINING THE "MASTER" ACTION SPACE ---
# We define specific atomic actions. 
# The agent will choose one index from this list.
# This covers Navigation, Combat, and Advanced Movement.

MASTER_ACTIONS = [
    [Button.MOVE_FORWARD],                                      # 0
    [Button.MOVE_BACKWARD],                                     # 1
    [Button.TURN_LEFT],                                         # 2
    [Button.TURN_RIGHT],                                        # 3
    [Button.MOVE_LEFT],                                         # 4 (Strafe)
    [Button.MOVE_RIGHT],                                        # 5 (Strafe)
    [Button.ATTACK],                                            # 6
    [Button.MOVE_FORWARD, Button.ATTACK],                       # 7 (Run & Gun)
    [Button.MOVE_FORWARD, Button.MOVE_LEFT, Button.ATTACK],     # 8 (Circle Strafe L)
    [Button.MOVE_FORWARD, Button.MOVE_RIGHT, Button.ATTACK],    # 9 (Circle Strafe R)
    [Button.MOVE_LEFT, Button.ATTACK],                          # 10
    [Button.MOVE_RIGHT, Button.ATTACK],                         # 11
    [Button.USE],                                               # 12 (Open Doors)
    [Button.SPEED],                                             # 13 (Sprint)
    [Button.SELECT_NEXT_WEAPON],                                # 14
    [Button.SELECT_PREV_WEAPON]                                 # 15
]

# --- 2. DEFINING THE MASKS ---
# 1 = Allowed, 0 = Banned
# We map the indices above to each scenario.

# Default (All Allowed)
MASK_ALL = np.ones(len(MASTER_ACTIONS), dtype=np.int8)

# Navigation Only (No Attack, No Weapon Change)
# Allowed: 0, 1, 2, 3, 4, 5, 12, 13 (Move, Turn, Strafe, Use, Sprint)
MASK_NAV = np.array([1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,0], dtype=np.int8)

# Simple Combat (No Use, No Weapon Change)
# Allowed: 0-11
MASK_COMBAT_BASIC = np.array([1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0], dtype=np.int8)

SCENARIO_CONFIGS = {
    "doom_basic": ("basic.cfg", MASK_COMBAT_BASIC),
    "doom_deadly_corridor": ("deadly_corridor.cfg", MASK_COMBAT_BASIC),
    "doom_defend_the_center": ("defend_the_center.cfg", MASK_COMBAT_BASIC),
    "doom_health_gathering_supreme": ("health_gathering_supreme.cfg", MASK_NAV),
    "doom_my_way_home": ("my_way_home.cfg", MASK_NAV),
    "doom_predict_position": ("predict_position.cfg", MASK_COMBAT_BASIC),
    "doom_take_cover": ("take_cover.cfg", MASK_NAV),
    "doom_deathmatch": ("deathmatch.cfg", MASK_ALL)  # <--- NEW SCENARIO
}

# --- 3. THE MASKING WRAPPER ---
class DoomActionMaskWrapper(gym.Wrapper):
    def __init__(self, env, action_mask):
        super().__init__(env)
        self.action_mask = action_mask
        
        # Update Observation Space to include the mask
        # We assume the original env has a 'obs' key (Dict space) or is a Box (Image)
        orig_space = self.env.observation_space
        
        new_spaces = {
            "action_mask": gym.spaces.Box(0, 1, shape=(len(MASTER_ACTIONS),), dtype=np.int8)
        }
        
        # If original was already a Dict, copy it. If it was just an Image (Box), wrap it.
        if isinstance(orig_space, gym.spaces.Dict):
            for k, v in orig_space.spaces.items():
                new_spaces[k] = v
        else:
            new_spaces["obs"] = orig_space
            
        self.observation_space = gym.spaces.Dict(new_spaces)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._add_mask(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._add_mask(obs)
        return obs, reward, terminated, truncated, info

    def _add_mask(self, obs):
        # Sample Factory expects the mask inside the dict
        # If obs is just an image, we wrap it
        if not isinstance(obs, dict):
            obs = {"obs": obs}
        
        obs["action_mask"] = self.action_mask
        return obs

# --- 4. ENVIRONMENT FACTORY ---
def make_multi_task_doom_env(full_env_name, cfg=None, env_config=None, render_mode=None, **kwargs):
    # Pick scenario
    import random
    scenario_name = random.choice(list(SCENARIO_CONFIGS.keys()))
    config_file, action_mask = SCENARIO_CONFIGS[scenario_name]
    
    # Create Spec with OUR custom Master Action Space
    # We pass the raw list of lists; Sample Factory/VizDoom Utils will parse it.
    env_spec = DoomSpec(scenario_name, config_file, MASTER_ACTIONS)
    
    # Create Base Env
    env = make_doom_env_from_spec(
        env_spec, 
        full_env_name, 
        cfg=cfg, 
        env_config=env_config, 
        render_mode=render_mode, 
        **kwargs
    )
    
    # Wrap it
    env = DoomActionMaskWrapper(env, action_mask)
    return env

def register_custom_components():
    register_env("doom_multi_task", make_multi_task_doom_env)
    # Important: Register standard VizDoom Encoder (or your Custom ViT if you use it)
    global_model_factory().register_encoder_factory(make_vizdoom_encoder)
