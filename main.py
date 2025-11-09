#!/usr/bin/env python3
"""
Script nivel 0:
- Controlar VizDoom vía API (sin teclado humano).
- Ver cómo se arma el loop: reset -> get_state -> choose_action -> make_action.
- Guardar logs simples para futuros experimentos / debugging.
"""

import vizdoom as vzd
import os
import random
from datetime import datetime
import time

DOOM_DIR = "/home/diegorandolp/Deep/Doom_project"

# -------------------------
# 1. Construir el entorno
# -------------------------

def get_config_path():
    """
    Ubica el archivo basic.cfg dentro de la instalación de VizDoom.
    Ajusta esto si tienes tus configs en otra carpeta.
    """
    return os.path.join(DOOM_DIR, "scenarios", "basic.cfg")

def create_game():
    """
    Crea y configura una instancia de DoomGame.
    En este nivel usamos:
    - Modo PLAYER para ver el juego en "tiempo real".
    - Ventana visible para confirmar visualmente que las acciones por API funcionan.
    """
    game = vzd.DoomGame()
    config_path = get_config_path()
    game.load_config(config_path)

    # Mostrar ventana (útil al inicio para debug visual).
    game.set_window_visible(True)

    # Modo:
    # - PLAYER: control paso a paso (ideal para entender el loop).
    # - SPECTATOR/ASYNC/etc. los veremos más adelante.
    game.set_mode(vzd.Mode.PLAYER)

    # Opcional: resolución de la pantalla (más chico = más rápido).
    # Puedes cambiar esto luego para tu encoder visual.
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Inicializar el juego (después de setear todo).
    game.init()
    return game


# -------------------------
# 2. Definir acciones
# -------------------------

def get_actions():
    """
    Para el escenario basic.cfg normalmente hay 3 botones:
    [TURN_LEFT, TURN_RIGHT, ATTACK]

    Cada acción es una lista de 0/1 indicando si se presiona cada botón.
    Aquí definimos algunas combinaciones útiles.
    """
    actions = [
        [1, 0, 0],  # girar a la izquierda
        [0, 1, 0],  # girar a la derecha
        [0, 0, 1],  # disparar
        [0, 0, 0],  # no hacer nada
        [1, 0, 1],  # girar izquierda + disparar
        [0, 1, 1],  # girar derecha + disparar
    ]
    return actions


# -------------------------
# 3. Loop principal
# -------------------------

def main():
    game = create_game()
    actions = get_actions()

    NUM_EPISODES = 5
    FRAME_SKIP = 4  # Avanza varios frames por acción (más rápido y más estable)

    # Directorio para logs simples
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join(
        "logs",
        f"basic_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    log_file = open(log_path, "w", encoding="utf-8")

    # Escribimos cabecera del log (para que luego sea fácil parsearlo)
    log_file.write("# frame\taction\treward\thealth-armor\n")

    for ep in range(NUM_EPISODES):
        print(f"\n[EPISODIO {ep + 1}]")
        game.new_episode()

        while not game.is_episode_finished():
            state = game.get_state()

            # state contiene:
            # - state.screen_buffer: imagen del juego (array)
            # - state.game_variables: por ejemplo salud, munición, etc. según el escenario
            # - state.number: índice del frame dentro del episodio
            frame_idx = state.number
            # Preferred: use the game API
            health = game.get_game_variable(vzd.GameVariable.HEALTH)
            selected_ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
            kill_count = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
            game_vars = [health, selected_ammo, kill_count]

            # ===== ACCIÓN POR API =====
            # Aquí va la "política".
            # Por ahora usamos algo aleatorio solo para probar el pipeline.
            # Luego reemplazamos esto por tu red neuronal.
            action = random.choice(actions)

            # Ejecutamos la acción.
            # FRAME_SKIP indica cuántos ticks del juego avanza manteniendo esa acción.
            reward = game.make_action(action, FRAME_SKIP)

            # ===== LOG =====
            # Guardamos la transición de forma muy simple.
            # Más adelante puedes serializar estados como npy/pt para entrenamiento offline.
            log_file.write(f"{frame_idx}\t{action}\t{reward:.3f}\t{game_vars}\n")

            # Print en consola para feedback rápido (sobrescribe la línea).
            print(
                f"frame={frame_idx:04d} action={action} "
                f"reward={reward:.3f} vars={game_vars}",
                end="\r",
            )

        # Al terminar el episodio vemos el score total
        total_reward = game.get_total_reward()
        print(f"\nFin del episodio {ep + 1} | Score total = {total_reward:.3f}")

    log_file.close()
    time.sleep(5)
    game.close()
    print(f"\nLogs guardados en: {log_path}")


if __name__ == "__main__":
    main()
