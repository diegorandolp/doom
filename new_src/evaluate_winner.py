"""
python evaluate_winner.py   --env=doom_multi_task   --experiment=doom_champion_v1   --algo=APPO   --train_dir=./train_dir   --force_scenario=doom_predict_position --max_num_episodes=5   --visualize --load_checkpoint_kind=best

"""



import sys
import os
import random
import torch

# Sample Factory Imports
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.enjoy import enjoy  # <--- Import the evaluation loop

# VizDoom Imports
from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
from sf_examples.vizdoom.doom.doom_utils import make_doom_env_from_spec, DoomSpec
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.doom.action_space import doom_action_space_extended

# --- CONFIGURATION ---
SCENARIO_CONFIGS = {
    "doom_basic": "basic.cfg",
    "doom_deadly_corridor": "deadly_corridor.cfg",
    "doom_defend_the_center": "defend_the_center.cfg",
    "doom_health_gathering_supreme": "health_gathering_supreme.cfg",
    "doom_my_way_home": "my_way_home.cfg",
    "doom_predict_position": "predict_position.cfg",
    "doom_take_cover": "take_cover.cfg"
}

def make_multi_task_doom_env(full_env_name, cfg=None, env_config=None, render_mode=None, **kwargs):
    # 1. Determine which scenario to play
    # If the user set an environment variable (via our script), force that scenario.
    # Otherwise, pick random (default behavior).
    forced_scenario = os.environ.get("FORCE_SCENARIO")
    
    if forced_scenario and forced_scenario in SCENARIO_CONFIGS:
        scenario_name = forced_scenario
    else:
        scenario_name = random.choice(list(SCENARIO_CONFIGS.keys()))

    config_file = SCENARIO_CONFIGS[scenario_name]
    
    # 2. specific 'render_mode' logic for visualization
    # If we are saving videos, we might need specific render modes depending on SF version,
    # but usually passing render_mode='rgb_array' is handled by the wrapper if save_video is on.

    env_spec = DoomSpec(scenario_name, config_file, doom_action_space_extended())
    
    return make_doom_env_from_spec(
        env_spec, 
        full_env_name, 
        cfg=cfg, 
        env_config=env_config, 
        render_mode=render_mode, 
        **kwargs
    )

def register_custom_components():
    register_env("doom_multi_task", make_multi_task_doom_env)
    global_model_factory().register_encoder_factory(make_vizdoom_encoder)

def parse_vizdoom_cfg(argv=None, evaluation=True):
    # Note evaluation=True here
    parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    
    # Add our custom flag for forcing a scenario
    parser.add_argument("--force_scenario", type=str, default=None, 
                        help="Force a specific scenario (e.g., doom_deadly_corridor) for evaluation")
    parser.add_argument("--visualize", action="store_true", 
                        help="Enable video recording")

    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg

def main():
    # 1. Register
    register_custom_components()

    # 2. Parse Config
    cfg = parse_vizdoom_cfg(evaluation=True)

    # 3. Handle Custom Flags
    if cfg.force_scenario:
        print(f"!!! FORCING SCENARIO: {cfg.force_scenario} !!!")
        os.environ["FORCE_SCENARIO"] = cfg.force_scenario

    if cfg.visualize:
        print("!!! VISUALIZATION ENABLED (Saving Video) !!!")
        # Sample Factory settings to enable video saving
        cfg.save_video = True
        cfg.video_frames = 2000  # Record 2000 frames (~1 minute)
        cfg.video_name_prefix = f"replay_{cfg.force_scenario or 'random'}"
        cfg.no_render = False # Headless mode (required for cluster)
    
    # 4. Run Evaluation
    status = enjoy(cfg)
    return status

if __name__ == "__main__":
    sys.exit(main())
