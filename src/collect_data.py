import vizdoom as vzd
import torch
import numpy as np
import cv2
import random
import os

# --- CONFIGURATION ---
# We only collect data for the "Core 5" to build a solid foundation
SCENARIOS_TO_COLLECT = [
    "basic", 
    "defend_the_center", 
    "health_gathering", 
    "defend_the_line", 
    "deadly_corridor"
]

# Total episodes per scenario (adjust based on speed, 200 is a good start)
EPISODES_PER_MAP = 200 

def get_random_action(available_buttons):
    """
    Generates a random valid action based on what buttons are available.
    """
    # Create a binary action list [0, 0, 0, ...]
    action = [0] * len(available_buttons)
    
    # 1. Always Pick a random button to press
    # (This is a 'twitchy' bot, good for generating diverse visual data)
    press_idx = random.randint(0, len(available_buttons) - 1)
    action[press_idx] = 1
    
    # 2. Heuristic Overrides (Make the data slightly better than random)
    # If we have "ATTACK" and random chance, press it (Aggression)
    if vzd.Button.ATTACK in available_buttons and random.random() < 0.3:
        # Find index of ATTACK
        idx = available_buttons.index(vzd.Button.ATTACK)
        action[idx] = 1

    # If we have "MOVE_FORWARD" and random chance, press it (Exploration)
    if vzd.Button.MOVE_FORWARD in available_buttons and random.random() < 0.6:
        idx = available_buttons.index(vzd.Button.MOVE_FORWARD)
        action[idx] = 1

    return action

def collect(scenario_name):
    print(f"--- Collecting: {scenario_name} ---")
    
    game = vzd.DoomGame()
    
    # Check if WAD exists standard path or local
    # VizDoom usually installs scenarios in the library path, or we look in ./scenarios
    # We try to load using the standard name which VizDoom resolves internally often
    try:
        game.load_config(os.path.join(vzd.scenarios_path, f"{scenario_name}.cfg"))
    except:
        # Fallback: assume user might have them in a local folder
        print(f"Could not find config in standard path, trying local ./scenarios/")
        game.load_config(f"scenarios/{scenario_name}.cfg")

    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(False)
    game.init()

    # Get the list of buttons this map actually uses (e.g., [MOVE_LEFT, ATTACK, ...])
    available_buttons = game.get_available_buttons()
    
    buffer_obs = []
    buffer_actions = []
    
    for ep in range(EPISODES_PER_MAP):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            
            # 1. Capture Frame
            frame = state.screen_buffer
            frame = cv2.resize(frame, (128, 72))
            frame = np.transpose(frame, (2, 0, 1)) # (C, H, W)
            
            # 2. Get Action (Robust to button definitions)
            action_cmd = get_random_action(available_buttons)
            game.make_action(action_cmd, 4)
            
            # 3. Save Data
            # Note: We need to map this specific action back to our 
            # "Universal Action Space" later, but for now we save the raw index 
            # of the button pressed or the raw command? 
            # Strategy: Save the 'action_cmd' directly. 
            # *CRITICAL*: Different maps have different vector lengths.
            # To fix this for a single dataset, we will padding later or 
            # just save valid buttons.
            # SIMPLIFICATION FOR COMPETITION:
            # We will save the index of the button pressed if it was a single press,
            # or just save the raw command if we use a specific mapping.
            
            # For this 'Quick & Dirty' Phase 1:
            # We will trust the RL phase to figure out the buttons.
            # Here we just want the FRAMES (Observations).
            # The Actions in BC are useful, but pixel understanding is the 80% win.
            # So we save the frame.
            
            buffer_obs.append(frame)
            
            # We save a dummy action or the raw command. 
            # Let's save the raw command, but pad it to 10 (max buttons usually)
            padded_action = action_cmd + [0] * (10 - len(action_cmd))
            buffer_actions.append(padded_action[:10])

        if (ep+1) % 50 == 0:
            print(f"  Episode {ep+1} done")

    game.close()
    
    # Convert to Tensors
    print(f"Saving {len(buffer_obs)} frames...")
    # Using uint8 for images saves massive RAM/Disk space (0-255)
    data_dict = {
        "observations": torch.tensor(np.array(buffer_obs), dtype=torch.uint8),
        "actions": torch.tensor(np.array(buffer_actions), dtype=torch.int8)
    }
    torch.save(data_dict, f"data/{scenario_name}_dataset.pt")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
        
    for scen in SCENARIOS_TO_COLLECT:
        try:
            collect(scen)
        except Exception as e:
            print(f"Skipping {scen} due to error: {e}")
            # This ensures one broken config doesn't stop the whole script
