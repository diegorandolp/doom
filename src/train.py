import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import glob
import gymnasium as gym # Changed from import gym
from gymnasium.vector import AsyncVectorEnv # Changed from gym.vector

# Local Imports
from model import DoomAgent
from wrappers import DoomEnv

# --- CRITICAL GPU CHECK ---
if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA not available! I am refusing to run on CPU. Check your drivers/installation.")

print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
torch.cuda.empty_cache() # Clear any residual memory
# --------------------------

# --- HYPERPARAMETERS (The Tuning Knobs) ---
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr_bc": 3e-4,          # Learning rate for cloning
    "lr_ppo": 1e-4,         # Lower learning rate for fine-tuning
    "gamma": 0.99,          # Discount factor
    "gae_lambda": 0.95,
    "ppo_clip": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,   # Keeps the agent exploring
    "num_envs": 16,         # Parallel environments (Adjust based on CPU cores)
    "rollout_steps": 128,   # Steps per env per update
    "ppo_epochs": 4,        # How many times to reuse data
    "batch_size": 2048,     # 40GB GPU can handle massive batches
    "hidden_size": 512,
    "bc_epochs": 5,         # How many passes over the fake data
    "total_timesteps": 50_000_000
}

# Scenario Mapping (Must match the IDs used in wrappers)
SCENARIOS = [
    "basic", "defend_the_center", "health_gathering", 
    "defend_the_line", "deadly_corridor"
]
SCENARIO_MAP = {name: i for i, name in enumerate(SCENARIOS)}

# --- HELPER: Action Converter ---
def buttons_to_action_idx(button_list):
    """
    Converts the raw button list [1,0,0...] back to our discrete index 0-6.
    Matches the logic in wrappers.py
    """
    # 0:Fwd, 1:Back, 2:Left, 3:Right, 4:Atk, 5:StrL, 6:StrR
    # This must match the 'btn_map' in wrappers.py exactly.
    if button_list[0] == 1: return 0 # Forward
    if button_list[1] == 1: return 1 # Backward
    if button_list[2] == 1: return 2 # Turn L
    if button_list[3] == 1: return 3 # Turn R
    if button_list[4] == 1: return 4 # Attack
    if len(button_list) > 5 and button_list[5] == 1: return 5 # Strafe L
    if len(button_list) > 6 and button_list[6] == 1: return 6 # Strafe R
    return 0 # Default to forward if messy

# --- PHASE 1: BEHAVIORAL CLONING (The Teacher) ---
def train_bc(agent, optimizer):
    print("--- Starting Phase 1: Behavioral Cloning ---")
    
    # 1. Load Data
    all_obs = []
    all_actions = []
    all_ids = []
    
    files = glob.glob("data/*_dataset.pt")
    if not files:
        print("CRITICAL WARNING: No data found in data/. Skipping BC.")
        return

    for f_path in files:
        print(f"Loading {f_path}...")
        data = torch.load(f_path, weights_only=False)
        obs = data["observations"] # (N, 3, 72, 128)
        acts_raw = data["actions"] # (N, num_buttons)
        
        # Determine Scenario ID from filename
        scen_name = os.path.basename(f_path).replace("_dataset.pt", "")
        scen_id = SCENARIO_MAP.get(scen_name, 0)
        
        # Convert raw buttons to discrete indices
        acts_discrete = [buttons_to_action_idx(a) for a in acts_raw.tolist()]
        
        all_obs.append(obs)
        all_actions.append(torch.tensor(acts_discrete, dtype=torch.long))
        all_ids.append(torch.full((len(obs),), scen_id, dtype=torch.long))

    # Concatenate all datasets
    full_obs = torch.cat(all_obs)
    full_actions = torch.cat(all_actions)
    full_ids = torch.cat(all_ids)
    
    dataset = TensorDataset(full_obs, full_actions, full_ids)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    loss_fn = nn.CrossEntropyLoss()
    
    # 2. Train Loop
    agent.train()
    for epoch in range(CONFIG["bc_epochs"]):
        total_loss = 0
        for batch_obs, batch_act, batch_id in loader:
            batch_obs, batch_act, batch_id = batch_obs.to(CONFIG["device"]), batch_act.to(CONFIG["device"]), batch_id.to(CONFIG["device"])
            
            # BC Limitation: We initialize Hidden State to 0 because data is shuffled
            # We are teaching the CNN to recognize enemies, not full temporal logic yet.
            hidden = torch.zeros(batch_obs.size(0), CONFIG["hidden_size"]).to(CONFIG["device"])
            
            optimizer.zero_grad()
            logits, _, _ = agent(batch_obs.float(), hidden, batch_id)
            
            loss = loss_fn(logits, batch_act)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"BC Epoch {epoch+1}: Avg Loss = {total_loss / len(loader):.4f}")
    
    print("--- BC Complete. Saving Warm-Start Model. ---")
    torch.save(agent.state_dict(), "models/bc_warmup.pth")

# --- PHASE 2: PPO TRAINING (The Athlete) ---
def make_env(rank):
    def _thunk():
        # Round Robin assignment of scenarios to workers
        # Worker 0 -> Map 0, Worker 1 -> Map 1, etc.
        scen_idx = rank % len(SCENARIOS)
        scen_name = SCENARIOS[scen_idx]
        env = DoomEnv(scen_name, scen_idx, visible=False)
        return env
    return _thunk

def train_ppo(agent, optimizer):
    print("--- Starting Phase 2: PPO Training ---")
    
    # 1. Setup Vector Environment
    envs = AsyncVectorEnv([make_env(i) for i in range(CONFIG["num_envs"])])
    
    # 2. Storage Setup
    obs_raw, _ = envs.reset()
    obs = torch.tensor(obs_raw, dtype=torch.uint8).to(CONFIG["device"])
    hidden = torch.zeros(CONFIG["num_envs"], CONFIG["hidden_size"]).to(CONFIG["device"])
    
    # Need to get scenario IDs for the current batch of envs
    # In our round-robin setup, this is static, but good to be dynamic
    # The wrapper returns info, but on reset/step we need to extract it.
    # For now, we calculate it based on index to save bandwidth.
    scen_ids = torch.tensor([i % len(SCENARIOS) for i in range(CONFIG["num_envs"])]).to(CONFIG["device"])

    global_step = 0
    
    while global_step < CONFIG["total_timesteps"]:
        # --- A. Rollout Collection ---
        batch_obs, batch_acts, batch_log_probs, batch_vals, batch_rews, batch_dones = [], [], [], [], [], []
        batch_hiddens = []
        
        for _ in range(CONFIG["rollout_steps"]):
            with torch.no_grad():
                # Forward pass
                # obs needs to be float for the network
                logits, vals, new_hidden = agent(obs.float(), hidden, scen_ids)
                
                # Action Selection (Stochastic)
                probs = torch.distributions.Categorical(logits=logits)
                action = probs.sample()
                log_prob = probs.log_prob(action)
            
            # Step Envs
            # Move to CPU for gym
            cpu_actions = action.cpu().numpy()
            next_obs, rewards, terminated, truncated, _ = envs.step(cpu_actions)
            dones = terminated | truncated # Combine them for PPO logic 
            # Store data
            batch_obs.append(obs)
            batch_hiddens.append(hidden)
            batch_acts.append(action)
            batch_log_probs.append(log_prob)
            batch_vals.append(vals)
            batch_rews.append(torch.tensor(rewards).to(CONFIG["device"]))
            batch_dones.append(torch.tensor(dones).to(CONFIG["device"]))
            
            # Update state
            obs = torch.tensor(next_obs, dtype=torch.uint8).to(CONFIG["device"])
            
            # Handle hidden state masking on done
            # If done, next hidden state should be 0 for that env
            mask = torch.tensor(1 - dones, dtype=torch.float32).unsqueeze(1).to(CONFIG["device"])
            hidden = new_hidden * mask
            
        # --- B. Advantage Calculation (GAE) ---
        # Get value of the very last state for bootstrapping
        with torch.no_grad():
            _, next_val, _ = agent(obs.float(), hidden, scen_ids)
            
        advantages = torch.zeros_like(torch.stack(batch_rews))
        last_gae_lam = 0
        
        # Reverse loop
        for t in reversed(range(CONFIG["rollout_steps"])):
            if t == CONFIG["rollout_steps"] - 1:
                nextnonterminal = 1.0 - batch_dones[t].float()
                nextvalues = next_val
            else:
                nextnonterminal = 1.0 - batch_dones[t].float()
                nextvalues = batch_vals[t+1]
                
            delta = batch_rews[t] + CONFIG["gamma"] * nextvalues * nextnonterminal - batch_vals[t]
            last_gae_lam = delta + CONFIG["gamma"] * CONFIG["gae_lambda"] * nextnonterminal * last_gae_lam
            advantages[t] = last_gae_lam
            
        returns = advantages + torch.stack(batch_vals)
        
        # Flatten the batch for PPO update
        # (Steps, Envs, ...) -> (Steps * Envs, ...)
        b_obs = torch.stack(batch_obs).view(-1, 3, 72, 128)
        b_acts = torch.stack(batch_acts).view(-1)
        b_log_probs = torch.stack(batch_log_probs).view(-1)
        b_returns = returns.view(-1)
        b_advs = advantages.view(-1)
        b_hiddens = torch.stack(batch_hiddens).view(-1, CONFIG["hidden_size"])
        # Scenario IDs need to be repeated for the time steps
        b_scen_ids = scen_ids.repeat(CONFIG["rollout_steps"])

        # Normalize advantages (Critical for PPO stability)
        b_advs = (b_advs - b_advs.mean()) / (b_advs.std() + 1e-8)
        
        # --- C. PPO Update ---
        dataset_len = b_obs.size(0)
        indexes = np.arange(dataset_len)
        
        for _ in range(CONFIG["ppo_epochs"]):
            np.random.shuffle(indexes)
            
            for start in range(0, dataset_len, CONFIG["batch_size"]):
                end = start + CONFIG["batch_size"]
                mb_idx = indexes[start:end]
                
                mb_obs = b_obs[mb_idx]
                mb_acts = b_acts[mb_idx]
                mb_old_log_probs = b_log_probs[mb_idx]
                mb_returns = b_returns[mb_idx]
                mb_advs = b_advs[mb_idx]
                mb_hidden = b_hiddens[mb_idx]
                mb_ids = b_scen_ids[mb_idx]
                
                # Re-evaluate logic (Forward pass)
                # Note: We detach hidden here because we are not backpropping 
                # through the whole episode history, just this step (Truncated BPTT approximation)
                new_logits, new_vals, _ = agent(mb_obs.float(), mb_hidden.detach(), mb_ids)
                
                new_probs = torch.distributions.Categorical(logits=new_logits)
                new_log_probs = new_probs.log_prob(mb_acts)
                entropy = new_probs.entropy().mean()
                
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # Surrogate Loss
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1.0 - CONFIG["ppo_clip"], 1.0 + CONFIG["ppo_clip"]) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = 0.5 * ((new_vals - mb_returns) ** 2).mean()
                
                loss = policy_loss + CONFIG["value_coef"] * value_loss - CONFIG["entropy_coef"] * entropy
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        if global_step % (CONFIG["batch_size"] * 100) == 0:
            # Calculate rough average reward of this batch (unscaled if possible, but scaled is fine)
            avg_reward = b_returns.mean().item()
            print(f"Step {global_step} | Avg Return: {avg_reward:.2f} | Value Loss: {value_loss.item():.1f}")

        
        global_step += CONFIG["num_envs"] * CONFIG["rollout_steps"]
        print(f"Step {global_step} | Policy Loss: {policy_loss.item():.3f} | Value Loss: {value_loss.item():.3f}")
        
        if global_step % 1_000_000 == 0:
            torch.save(agent.state_dict(), f"models/ppo_step_{global_step}.pth")

# --- MAIN ---
if __name__ == "__main__":
    # Create Models folder
    if not os.path.exists("models"): os.makedirs("models")
    
    # Initialize Agent
    agent = DoomAgent(action_space_size=7, num_scenarios=len(SCENARIOS)).to(CONFIG["device"])
    
    # Optimizer
    optimizer = optim.Adam(agent.parameters(), lr=CONFIG["lr_bc"])
    
    # 1. Warm Start (BC)
    # Check if we already have a BC model to skip this if restarting
    if not os.path.exists("models/bc_warmup.pth"):
        train_bc(agent, optimizer)
    else:
        print("Loading existing BC weights...")
        agent.load_state_dict(torch.load("models/bc_warmup.pth"))

    # 2. RL Training (PPO)
    # Switch learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = CONFIG["lr_ppo"]
        
    train_ppo(agent, optimizer)
