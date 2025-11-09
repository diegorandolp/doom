#!/usr/bin/env python3
"""
Wrapper estilo Gym para VizDoom.

Objetivo:
- Tener una interfaz estándar: reset(), step(), render(), close().
- Conectarla fácilmente con librerías de RL (propias o tipo Stable-Baselines3).

Notas:
- Usa modo headless por defecto (sin ventana).
- Acción = entero en [0, N_ACTIONS).
- Obs = frame RGB (H, W, 3) como uint8.
"""

import os
import numpy as np
import vizdoom as vzd
import gymnasium as gym
from gymnasium import spaces

class VizDoomEnv(gym.Env):
    """
    Env genérico para escenarios tipo combate (TURN_LEFT, TURN_RIGHT, ATTACK).

    Parámetros:
    - doom_dir: carpeta donde están los .cfg y .wad
    - scenario: nombre base del escenario (sin extensión)
    - frame_skip: cuántos ticks avanza por acción
    - render: si True, muestra ventana (útil solo para debug manual)
    """

    metadata = {"render_modes": ["human"], "render_fps": 35}

    def __init__(
            self,
            doom_dir="/home/diegorandolp/Deep/Doom_project",
            scenario="defend_the_center",
            frame_skip=4,
            render=False,
    ):
        super().__init__()

        self.doom_dir = doom_dir
        self.scenario = scenario
        self.frame_skip = frame_skip
        self.render_enabled = render

        # ---- Definir acciones discretas ----
        # [TURN_LEFT, TURN_RIGHT, ATTACK]
        self.actions = [
            [1, 0, 0],  # 0: girar izquierda
            [0, 1, 0],  # 1: girar derecha
            [0, 0, 1],  # 2: disparar
            [0, 0, 0],  # 3: nada
            [1, 0, 1],  # 4: izquierda + disparar
            [0, 1, 1],  # 5: derecha + disparar
        ]
        self.action_space = spaces.Discrete(len(self.actions))

        # ---- Crear juego interno ----
        self.game = vzd.DoomGame()
        self._load_config()
        self._set_mode_and_window()

        # Inicializar VizDoom
        self.game.init()

        # Definir observation_space basado en screen_buffer
        # Usamos RGB 320x240 por defecto (ajustable en _set_mode_and_window)
        h, w, c = 240, 320, 3
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(h, w, c),
            dtype=np.uint8,
        )

        # Para llevar control del último frame por si el episodio termina
        self.last_obs = None

    # ---------------------
    # Configuración interna
    # ---------------------

    def _load_config(self):
        """
        Carga el .cfg del escenario.
        Si no existe uno específico, intenta basic.cfg como fallback.
        """
        cfg_name = f"{self.scenario}.cfg"
        cfg_path = os.path.join(self.doom_dir, "scenarios", cfg_name)

        if not os.path.isfile(cfg_path):
            print(f"[WARN] No se encontró {cfg_path}, usando basic.cfg")
            cfg_path = os.path.join(self.doom_dir, "scenarios", "basic.cfg")

        self.game.load_config(cfg_path)

        # Setear .wad explícito si existe
        wad_path = os.path.join(self.doom_dir, "scenarios", f"{self.scenario}.wad")
        if os.path.isfile(wad_path):
            self.game.set_doom_scenario_path(wad_path)

    def _set_mode_and_window(self):
        """
        Configura modo de juego, ventana, resolución, etc.
        """
        # Ventana visible solo si render=True
        self.game.set_window_visible(self.render_enabled)

        # Sin sonido para entrenamiento
        self.game.set_sound_enabled(False)

        # Resolución bajita para acelerar entrenamiento
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)

        # Modo síncrono controlado por agente
        self.game.set_mode(vzd.Mode.PLAYER)

    # ---------------------
    # API Gym
    # ---------------------

    def reset(self, *, seed=None, options=None):
        """
        Reset del episodio.
        Devuelve:
        - obs: frame inicial (uint8)
        - info: dict con variables útiles (health, ammo, kills, etc.)
        """
        if seed is not None:
            # VizDoom también permite seeds, pero aquí lo manejas externamente si quieres
            super().reset(seed=seed)

        self.game.new_episode()
        obs = self._get_observation()
        info = self._get_info()

        self.last_obs = obs
        # Gymnasium: return obs, info
        return obs, info

    def step(self, action):
        """
        Ejecuta 1 acción (índice entero) durante frame_skip ticks.
        Devuelve:
        - obs
        - reward
        - terminated (fin por muerte/objetivo)
        - truncated (no lo usamos aquí, va en False)
        - info
        """
        # Mapear índice a vector de botones
        if isinstance(action, np.ndarray):
            action = int(action.item())
        action = int(action)
        assert self.action_space.contains(action), f"Acción inválida: {action}"

        button_action = self.actions[action]

        reward = self.game.make_action(button_action, self.frame_skip)

        done = self.game.is_episode_finished()

        if done:
            # Cuando termina el episodio, VizDoom ya no tiene screen_buffer válido.
            obs = self.last_obs
        else:
            obs = self._get_observation()
            self.last_obs = obs

        info = self._get_info()

        # Terminación:
        terminated = done   # por ahora todo done es terminal
        truncated = False   # aquí no aplicamos corte por tiempo, pero puedes agregarlo

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Renderizar solo si se configuró con render=True.
        Si quieres verlo, crea el env con render=True para que VizDoom muestre la ventana.
        """
        # VizDoom maneja la ventana internamente cuando set_window_visible(True),
        # así que aquí no necesitamos hacer mucho.
        pass

    def close(self):
        if self.game is not None:
            self.game.close()

    # ---------------------
    # Helpers internos
    # ---------------------

    def _get_observation(self):
        """
        Obtiene el frame actual como imagen RGB uint8.
        """
        state = self.game.get_state()
        if state is None:
            # Puede pasar justo al terminar el episodio; devolvemos último obs si existe
            if self.last_obs is not None:
                return self.last_obs
            # Si no hay nada, devolvemos negro
            h, w, c = self.observation_space.shape
            return np.zeros((h, w, c), dtype=np.uint8)

        screen = state.screen_buffer  # shape (C, H, W) o (H, W) según config

        if screen is None:
            h, w, c = self.observation_space.shape
            return np.zeros((h, w, c), dtype=np.uint8)

        # Convertir a (H, W, C)
        if len(screen.shape) == 2:
            # escala de grises -> expandimos a 3 canales
            screen = np.stack([screen] * 3, axis=-1)
        elif len(screen.shape) == 3:
            # (C, H, W) -> (H, W, C)
            screen = np.transpose(screen, (1, 2, 0))
        else:
            raise ValueError(f"Forma de pantalla inesperada: {screen.shape}")

        # Asegurar tipo uint8
        if screen.dtype != np.uint8:
            screen = screen.astype(np.uint8)

        # Si difiere de la shape esperada, puedes hacer resize aquí con cv2 o similar.
        # Por simplicidad asumimos que coincide con RES_320X240.
        return screen

    def _get_info(self):
        """
        Retorna variables útiles para debugging/analítica.
        Esto no afecta el entrenamiento estándar, pero sirve mucho.
        """
        info = {}
        if self.game.is_episode_finished():
            return info

        gv = vzd.GameVariable

        def safe_var(var):
            try:
                return int(self.game.get_game_variable(var))
            except:
                return 0

        info["health"] = safe_var(gv.HEALTH)
        info["ammo"] = safe_var(gv.SELECTED_WEAPON_AMMO)
        info["kills"] = safe_var(gv.KILLCOUNT)

        return info


# -------------------------
# Ejemplo de uso standalone
# -------------------------

if __name__ == "__main__":
    env = VizDoomEnv(
        doom_dir="/home/diegorandolp/Deep/Doom_project",
        scenario="defend_the_center",
        frame_skip=4,
        render=True,  # pon True si quieres ver la ventana
    )

    num_episodes = 3

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Política random como test de integración
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            done = terminated or truncated

        print(f"Episodio {ep + 1} terminado | Reward total: {total_reward:.3f}")

    env.close()
