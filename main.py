#!/usr/bin/env python3
"""
Script nivel 0 (headless + policy tonta):
- Controlar VizDoom vía API sin ventana (modo entrenamiento).
- Loop: reset -> get_state -> choose_action -> make_action.
- Policy tonta pero lógica para escenarios tipo defend_the_center.
- Guardar logs simples (frame, acción, reward, [health, ammo, kills]).
"""
import time
import vizdoom as vzd
import os
from datetime import datetime

# Ruta base donde tienes los escenarios de VizDoom
DOOM_DIR = "/home/diegorandolp/Deep/Doom_project"

# Escenario elegido:
# opciones típicas: "basic", "defend_the_center", "defend_the_line", "deathmatch"
SCENARIO = "defend_the_center"

# -------------------------
# 1. Helpers de config
# -------------------------

def get_config_path():
    """
    Devuelve la ruta al archivo .cfg del escenario elegido.

    Convención usual de VizDoom:
    - basic.cfg
    - defend_the_center.cfg
    - defend_the_line.cfg
    - deathmatch.cfg

    Ajusta si tu estructura es distinta.
    """
    cfg_name = f"{SCENARIO}.cfg"
    cfg_path = os.path.join(DOOM_DIR, "scenarios", cfg_name)

    if not os.path.isfile(cfg_path):
        # Fallback: si no existe, usa basic.cfg para no romper,
        # pero idealmente siempre deberías tener el cfg del escenario.
        print(f"[WARN] No se encontró {cfg_path}, usando basic.cfg")
        cfg_path = os.path.join(DOOM_DIR, "scenarios", "basic.cfg")

    return cfg_path


def create_game():
    """
    Crea y configura DoomGame para correr SIN ventana.
    Ideal para entrenamiento en servidor/cluster.
    """
    game = vzd.DoomGame()

    # Config principal del escenario
    config_path = get_config_path()
    game.load_config(config_path)

    # Escenario (.wad) asociado (solo si aplica).
    # Si el cfg ya lo setea internamente, esto es opcional.
    wad_path = os.path.join(DOOM_DIR, "scenarios", f"{SCENARIO}.wad")
    if os.path.isfile(wad_path):
        game.set_doom_scenario_path(wad_path)

    # -------- CAMBIOS CLAVE PARA MODO SIN VENTANA --------
    # 1) No mostrar ventana.
    game.set_window_visible(True)

    # 2) Opcional: sin sonido (ligeramente más rápido).
    game.set_sound_enabled(False)

    # Modo de control:
    # PLAYER = step síncrono controlado por nosotros (perfecto para RL).
    game.set_mode(vzd.Mode.PLAYER)

    # Resolución: más chico = más rápido.
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)

    game.init()
    return game


# -------------------------
# 2. Definir acciones
# -------------------------

def get_actions():
    """
    Definimos acciones para escenarios de combate simples con:
    [TURN_LEFT, TURN_RIGHT, ATTACK]
    """
    return [
        [1, 0, 0],  # 0: girar a la izquierda
        [0, 1, 0],  # 1: girar a la derecha
        [0, 0, 1],  # 2: disparar
        [0, 0, 0],  # 3: no hacer nada
        [1, 0, 1],  # 4: girar izquierda + disparar
        [0, 1, 1],  # 5: girar derecha + disparar
    ]


# -------------------------
# 3. Policy tonta pero lógica
# -------------------------

def heuristic_policy(actions, health, ammo, kills, prev_kills, step_idx):
    """
    Política handcrafted muy simple para escenarios tipo defend_the_center:

    Idea:
    - Si no tienes munición: deja de disparar y solo rota para "buscar".
    - Si tienes munición:
        - Mantén fuego frecuente (porque el enemigo está entrando al área).
        - Oscila derecha/izquierda para cubrir 360°.
    - Si ves que aumentan los kills (kills > prev_kills):
        - Sigue disparando en la misma dirección (algo funcionó).
    - Si pasó mucho tiempo sin kills:
        - Cambia de dirección.

    No es inteligente, pero:
    - Usa info del entorno.
    - Genera comportamiento consistente y un poco menos random.
    """

    # Indices legibles
    TURN_LEFT = 0
    TURN_RIGHT = 1
    ATTACK = 2
    LEFT_SHOOT = 4
    RIGHT_SHOOT = 5

    # Sin munición → gira sin disparar (barrer entorno)
    if ammo <= 0:
        # alterna direcciones según step_idx para no quedarse pegado
        if (step_idx // 10) % 2 == 0:
            return actions[TURN_LEFT]
        else:
            return actions[TURN_RIGHT]

    # Con munición:
    # Si acabamos de matar a alguien → insiste un poco en la misma rotación + disparo
    if kills > prev_kills:
        # Probamos girar derecha + disparar (o podrías memorizar última dirección)
        return actions[RIGHT_SHOOT]

    # Si no matas hace rato: oscilar
    # Ejemplo: bloques de 20 steps, 10 mirando-disparando a la izquierda, 10 a la derecha
    phase = (step_idx // 10) % 4

    if phase in [0, 1]:
        # Disparar girando izquierda
        return actions[LEFT_SHOOT]
    else:
        # Disparar girando derecha
        return actions[RIGHT_SHOOT]


# -------------------------
# 4. Loop principal
# -------------------------

def main():
    game = create_game()
    actions = get_actions()

    NUM_EPISODES = 5
    FRAME_SKIP = 4  # Avanza varios ticks con la misma acción

    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join(
        "logs",
        f"{SCENARIO}_heuristic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write("# frame\taction\treward\thealth\tammo\tkills\n")

    for ep in range(NUM_EPISODES):
        print(f"\n[EPISODIO {ep + 1}]")
        game.new_episode()

        prev_kills = 0
        step_idx = 0

        while not game.is_episode_finished():
            time.sleep(0.5)
            state = game.get_state()
            frame_idx = state.number

            # Variables importantes del HUD
            health = game.get_game_variable(vzd.GameVariable.HEALTH)
            ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
            kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)

            # Policy tonta pero lógica
            action = heuristic_policy(
                actions=actions,
                health=health,
                ammo=ammo,
                kills=kills,
                prev_kills=prev_kills,
                step_idx=step_idx
            )

            reward = game.make_action(action, FRAME_SKIP)

            # Log
            log_file.write(
                f"{frame_idx}\t{action}\t{reward:.3f}\t{health}\t{ammo}\t{kills}\n"
            )

            # Feedback en consola
            print(
                f"ep={ep+1} step={step_idx:04d} frame={frame_idx:04d} "
                f"act={action} rew={reward:.3f} H={health} A={ammo} K={kills}",
                end="\r"
            )

            prev_kills = kills
            step_idx += 1

        total_reward = game.get_total_reward()
        print(f"\nFin del episodio {ep + 1} | Score total = {total_reward:.3f}")

    log_file.close()
    game.close()
    print(f"\nLogs guardados en: {log_path}")


if __name__ == "__main__":
    main()
