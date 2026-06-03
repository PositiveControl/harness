# Build task: top-down GTA-style game

Build a small top-down 2D Grand-Theft-Auto-style driving game in **Python using
pygame**, in this directory. Entry point must be `main.py` runnable with
`python main.py`. Keep all game code importable without launching a window
(guard the loop under `if __name__ == "__main__":`).

The game must run **headless-testable**: it must import and initialize cleanly when
the environment sets `SDL_VIDEODRIVER=dummy` (no real display).

Implement these milestones in order. Each is graded independently, so a partial
result still counts — do as many as you can, correctly.

- **M1 — window + loop**: open a game window, run a main loop, exit cleanly on quit.
- **M2 — player car**: a car sprite renders and is controllable — WASD / arrow keys
  accelerate, brake, and steer (rotate) the car.
- **M3 — world**: a top-down tiled map / road network larger than the viewport; the
  camera scrolls to follow the player car.
- **M4 — physics**: acceleration, steering, and friction — momentum, not instant
  teleport-style movement.
- **M5 — NPC traffic**: at least one AI-driven vehicle that moves around the map on
  its own.
- **M6 — collision**: collision detection and resolution between the player car and
  the world and/or NPC vehicles.
- **M7 — objective**: a mission / score loop (reach a waypoint, pick up a target, or a
  wanted-level mechanic) plus an on-screen HUD showing score or state.

Constraints:

- Pure Python + `pygame` only. No network calls, no asset downloads — generate any
  needed shapes/sprites in code.
- Single self-contained project rooted here. `main.py` is the entry point.
- Favor correctness and runnability over scope: a clean M1–M4 beats a crashing M1–M7.
