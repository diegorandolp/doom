import sys
import random
import torch

# Sample Factory Imports
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

# VizDoom Specific Imports
from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
from sf_examples.vizdoom.doom.doom_utils import make_doom_env_from_spec, DoomSpec
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
# Added this import based on your example to ensure correct buttons are available
#from sf_examples.vizdoom.doom.action_space import doom_action_space_extended 
from sf_examples.vizdoom.doom.action_space import doom_action_space_discretized

# --- CONFIGURATION: Mapping Scenarios to Config Files ---
SCENARIO_CONFIGS = {
    "doom_basic": "basic.cfg",
    "doom_deadly_corridor": "deadly_corridor.cfg",
    "doom_defend_the_center": "defend_the_center.cfg",
    "doom_health_gathering_supreme": "health_gathering_supreme.cfg",
    "doom_my_way_home": "my_way_home.cfg",
    #"doom_predict_position": "predict_position.cfg",
    #"doom_take_cover": "take_cover.cfg"
}

def make_multi_task_doom_env(full_env_name, cfg=None, env_config=None, render_mode=None, **kwargs):
    """
    Creates a Doom environment with a random scenario selected from the list.
    """
    # 1. Pick a random scenario name
    scenario_name = random.choice(list(SCENARIO_CONFIGS.keys()))
    
    # 2. Get the specific config file
    config_file = SCENARIO_CONFIGS[scenario_name]
    
    # 3. Create the Spec
    # We use the extended action space to ensure the agent can perform all moves (turn, shoot, strafe)
    # regardless of which map it is currently playing.
    env_spec = DoomSpec(scenario_name, config_file, doom_action_space_discretized())
    
    # 4. Create the environment
    # FIX: We now pass 'full_env_name' as the second argument, which fixes the TypeError.
    return make_doom_env_from_spec(
        env_spec, 
        full_env_name,  # <--- This was missing!
        cfg=cfg, 
        env_config=env_config, 
        render_mode=render_mode, 
        **kwargs
    )

def register_custom_components():
    """Register the custom environment AND the VizDoom model encoder."""
    register_env("doom_multi_task", make_multi_task_doom_env)
    global_model_factory().register_encoder_factory(make_vizdoom_encoder)

def parse_vizdoom_cfg(argv=None, evaluation=False):
    """
    Parses configuration arguments using the exact logic from sf_examples.
    """
    parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg

def verify_gpu_or_die():
    """Strict verification to ensure GPU is accessible."""
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
