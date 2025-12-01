import gymnasium as gym  # CHANGED: gym -> gymnasium
import vizdoom as vzd
import numpy as np
import cv2
import os

# Universal Buttons that map to our Agent's outputs
ACTIONS = [
    [0,0,0,0,0,0], # No-Op
    [1,0,0,0,0,0], # Move Forward
    [0,1,0,0,0,0], # Move Backward
    [0,0,1,0,0,0], # Turn Left
    [0,0,0,1,0,0], # Turn Right
    [0,0,0,0,1,0], # Strafe Left
    [0,0,0,0,0,1], # Strafe Right
]

class DoomEnv(gym.Env):
    def __init__(self, scenario_name, scenario_id_int, visible=False):
        super().__init__()
        self.scenario_id = scenario_id_int
        
        self.game = vzd.DoomGame()
        
        # Robust config loading
        config_path = os.path.join(vzd.scenarios_path, f"{scenario_name}.cfg")
        if not os.path.exists(config_path):
             config_path = f"scenarios/{scenario_name}.cfg"
             
        self.game.load_config(config_path)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_window_visible(visible)
        
        # Optimization: Disable HUD
        self.game.set_render_hud(False) 
        self.game.set_render_crosshair(False)
        self.game.set_render_decals(False)
        self.game.set_render_particles(False)
        
        self.game.init()
        
        self.available_buttons = self.game.get_available_buttons()
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(0, 255, (3, 72, 128), dtype=np.uint8)

    def step(self, action_idx):
        # ... Mapping logic ...
        cmd = [0] * len(self.available_buttons)
        btn_map = {
            0: vzd.Button.MOVE_FORWARD,
            1: vzd.Button.MOVE_BACKWARD,
            2: vzd.Button.TURN_LEFT,
            3: vzd.Button.TURN_RIGHT,
            4: vzd.Button.ATTACK,
            5: vzd.Button.MOVE_LEFT,
            6: vzd.Button.MOVE_RIGHT
        }
        
        target_btn = btn_map.get(action_idx)
        if target_btn in self.available_buttons:
            cmd[self.available_buttons.index(target_btn)] = 1
        
        reward = self.game.make_action(cmd, 4)
        reward = reward * 0.01
        done = self.game.is_episode_finished()
        
        if done:
            state = np.zeros((3, 72, 128), dtype=np.uint8)
        else:
            state = self.game.get_state().screen_buffer
            state = cv2.resize(state, (128, 72))
            state = np.transpose(state, (2, 0, 1))
            
        # GYMNASIUM CHANGE: Must return (obs, reward, terminated, truncated, info)
        terminated = done
        truncated = False # VizDoom usually doesn't have time limits handled by us
        
        return state, reward, terminated, truncated, {"scenario_id": self.scenario_id}

    def reset(self, seed=None, options=None):
        # GYMNASIUM CHANGE: Handle seed
        super().reset(seed=seed)
        
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        state = cv2.resize(state, (128, 72))
        state = np.transpose(state, (2, 0, 1))
        
        # GYMNASIUM CHANGE: Must return (obs, info)
        return state, {"scenario_id": self.scenario_id}
        
    def close(self):
        self.game.close()
