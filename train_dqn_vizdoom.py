#!/usr/bin/env python3
"""
Entrenamiento DQN básico en VizDoom con PyTorch.

Requisitos:
- vizdoom_gym_env.py en el mismo directorio.
- PyTorch instalado.
- Escenario defend_the_center (o basic) disponible.

Este script:
- Crea el entorno VizDoomEnv (gym-like).
- Define una red convolucional simple.
- Implementa DQN con:
    - Replay Buffer
    - Epsilon-greedy
    - Target network
- Entrena por N episodios y muestra el reward promedio.

Este es un baseline pedagógico: simple, directo, perfecto para iterar.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from vizdoom_gym_env import VizDoomEnv


# =========================
# 1. Hiperparámetros
# =========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPISODES = 2000          # súbelo luego
MAX_STEPS_PER_EPISODE = 2000
BATCH_SIZE = 64
GAMMA = 0.99 # factor that promotes long-term rewards over short-term rewards
LR = 1e-4

REPLAY_CAPACITY = 100_000
START_TRAINING_AFTER = 10_000      # número mínimo de transiciones en buffer
TRAIN_EVERY = 4                    # cada cuántos steps entrenar
TARGET_UPDATE_EVERY = 1_000        # cada cuántos steps copiar a la target net

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 200_000          # pasos para ir de 1.0 a 0.05

# Dimensiones objetivo para la red (estilo Atari)
OBS_H = 84
OBS_W = 84


# =========================
# 2. Utils: procesar observaciones
# =========================

def preprocess_obs(obs):
    """
    Convierte obs (H, W, 3) uint8 en:
    - escala de grises
    - tamaño (1, 84, 84) como float32 en [0,1]
    """
    # obs: np.array uint8 (H, W, C)
    if obs is None:
        return np.zeros((1, OBS_H, OBS_W), dtype=np.float32)

    # Promedio canales para gris
    gray = obs.mean(axis=2)

    # Resize sencillo con slicing/interpolación nearest (para no depender de cv2).
    # Asumimos obs ~ 240x320; hacemos un resize naive:
    h, w = gray.shape
    scale_y = h / OBS_H
    scale_x = w / OBS_W

    resized = np.zeros((OBS_H, OBS_W), dtype=np.float32)
    for i in range(OBS_H):
        for j in range(OBS_W):
            y = int(i * scale_y)
            x = int(j * scale_x)
            if y >= h:
                y = h - 1
            if x >= w:
                x = w - 1
            resized[i, j] = gray[y, x]

    # Normalizar a [0,1]
    resized /= 255.0

    # Añadir canal (1, H, W)
    return resized[np.newaxis, :, :].astype(np.float32)


# =========================
# 3. Replay Buffer
# =========================
# train on batched formed by random experiences
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    # save tuple state, action, reward, next_state, done
    def push(self, s, a, r, ns, d):
        # Guardamos como np.array para ahorrar memoria
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.stack(s),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(ns),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# =========================
# 4. Red Q (CNN estilo Atari)
# =========================

class CnnDQN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        # Input: (1, 84, 84)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)   # -> (32, 20, 20)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # -> (64, 9, 9)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # -> (64, 7, 7)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        # x: (B, 1, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Q-values por acción


# =========================
# 5. Epsilon scheduling
# =========================
# epsilon is like beta in b-VAE but it balances exploration(random action)/exploitation(best action learned)
# it starts with exploration and it decays to exploit learned strategies
def get_epsilon(step):
    """
    Decae linealmente desde EPS_START hasta EPS_END en EPS_DECAY_STEPS.
    Luego se mantiene en EPS_END.
    """
    if step >= EPS_DECAY_STEPS:
        return EPS_END
    frac = step / float(EPS_DECAY_STEPS)
    return EPS_START + frac * (EPS_END - EPS_START)


# =========================
# 6. Función de entrenamiento
# =========================

def train_step(policy_net, target_net, optimizer, replay_buffer):
    if len(replay_buffer) < BATCH_SIZE:
        return 0.0

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    states = torch.from_numpy(states).to(DEVICE)          # (B, 1, 84, 84)
    next_states = torch.from_numpy(next_states).to(DEVICE)
    actions = torch.from_numpy(actions).to(DEVICE)        # (B,)
    rewards = torch.from_numpy(rewards).to(DEVICE)        # (B,)
    dones = torch.from_numpy(dones).to(DEVICE)            # (B,)

    # Q(s, a) , input: states, output: Q-values (expected sum of future rewards)
    q_values = policy_net(states)                         # (B, num_actions)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Q_target = r + gamma * max_a' Q_target(s', a') * (1 - done)
    with torch.no_grad():
        # target_net is the same as policy_net except this one updates less frequently
        # it is frozen to keep the target fixed, otherwise the target would
        # be moving
        next_q_values = target_net(next_states).max(1)[0]
        # target is the ideal reward from the current state: current reward plus the optimal q value for
        # the next state (the best action in that state to maximize rewards)
        target = rewards + (1.0 - dones) * GAMMA * next_q_values

    loss = F.smooth_l1_loss(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()

    return loss.item()


# =========================
# 7. Main loop entrenamiento
# =========================

def main():
    env = VizDoomEnv(
        doom_dir="/home/diegorandolp/Deep/Doom_project",
        scenario="defend_the_center",
        frame_skip=4,
        render=False,
    )

    num_actions = env.action_space.n

    policy_net = CnnDQN(num_actions).to(DEVICE)
    target_net = CnnDQN(num_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(REPLAY_CAPACITY)

    global_step = 0
    episode_rewards = []

    for ep in range(1, NUM_EPISODES + 1):
        obs, info = env.reset()
        state = preprocess_obs(obs)
        episode_reward = 0.0

        for step in range(MAX_STEPS_PER_EPISODE):
            global_step += 1
            epsilon = get_epsilon(global_step)

            # Epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_tensor = torch.from_numpy(state).unsqueeze(0).to(DEVICE)  # (1,1,84,84)
                    q_vals = policy_net(s_tensor)
                    action = int(q_vals.argmax(dim=1).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state = preprocess_obs(next_obs)

            replay_buffer.push(state, action, reward, next_state, float(done))

            state = next_state
            episode_reward += reward

            # Entrenamiento periódico
            if global_step > START_TRAINING_AFTER and global_step % TRAIN_EVERY == 0:
                loss = train_step(policy_net, target_net, optimizer, replay_buffer)

            # Actualizar target net
            if global_step % TARGET_UPDATE_EVERY == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        episode_rewards.append(episode_reward)

        # Logging simple
        if ep % 10 == 0:
            avg_last_10 = np.mean(episode_rewards[-10:])
            print(
                f"[EP {ep}/{NUM_EPISODES}] "
                f"Reward ep={episode_reward:.1f} | "
                f"Avg10={avg_last_10:.1f} | "
                f"Buffer={len(replay_buffer)} | "
                f"Eps={epsilon:.3f}"
            )

    env.close()
    print("Entrenamiento terminado.")


if __name__ == "__main__":
    main()
