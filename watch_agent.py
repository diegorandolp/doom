#!/usr/bin/env python3
"""
Script para visualizar un agente DQN entrenado jugando VizDoom.
Carga los pesos .pth y renderiza el juego.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vizdoom_gym_env import VizDoomEnv

# ==========================================
# IMPORTANTE: Debe ser la misma arquitectura
# usada en el entrenamiento.
# (Idealmente, mueve esto a un archivo common.py)
# ==========================================

OBS_H = 84
OBS_W = 84

class CnnDQN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def preprocess_obs(obs):
    if obs is None:
        return np.zeros((1, OBS_H, OBS_W), dtype=np.float32)
    gray = obs.mean(axis=2)
    h, w = gray.shape
    scale_y = h / OBS_H
    scale_x = w / OBS_W
    resized = np.zeros((OBS_H, OBS_W), dtype=np.float32)
    for i in range(OBS_H):
        for j in range(OBS_W):
            y = int(i * scale_y)
            x = int(j * scale_x)
            if y >= h: y = h - 1
            if x >= w: x = w - 1
            resized[i, j] = gray[y, x]
    resized /= 255.0
    return resized[np.newaxis, :, :].astype(np.float32)

# ==========================================
# Configuración de Visualización
# ==========================================

MODEL_PATH = "dqn_vizdoom_best.pth" # Asegúrate que este archivo exista
SCENARIO = "defend_the_center"      # El mismo usado en training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def watch():
    print(f"Cargando modelo desde {MODEL_PATH}...")

    # 1. Crear entorno con render=True
    env = VizDoomEnv(
        doom_dir="/home/diegorandolp/Deep/Doom_project",
        scenario=SCENARIO,
        frame_skip=4,
        render=True  # <--- ESTO ACTIVA LA VENTANA
    )

    # 2. Instanciar modelo y cargar pesos
    num_actions = env.action_space.n
    policy_net = CnnDQN(num_actions).to(DEVICE)

    try:
        # map_location asegura que cargue en CPU si entrenaste en GPU pero visualizas en laptop
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        policy_net.eval() # Importante: pone la red en modo inferencia (cierra dropout/batchnorm si hubiera)
        print("Modelo cargado exitosamente.")
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo {MODEL_PATH}. Entrena primero.")
        return

    num_episodes = 5

    for ep in range(num_episodes):
        obs, info = env.reset()
        state = preprocess_obs(obs)
        done = False
        total_reward = 0

        print(f"Iniciando Episodio {ep+1}...")

        while not done:
            # Preprocesar input
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(DEVICE)

            # Seleccionar la MEJOR acción (Greedy)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = int(q_values.argmax(dim=1).item())

            # Ejecutar
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = preprocess_obs(obs)

            total_reward += reward

            # Pequeña pausa para que el ojo humano pueda seguir la acción
            # VizDoom va muy rápido sin esto.
            time.sleep(0.05)

        print(f"Episodio finalizado. Reward Total: {total_reward}")
        time.sleep(1.0) # Pausa entre episodios

    env.close()

if __name__ == "__main__":
    watch()