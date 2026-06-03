# Build task: top-down GTA2-style web game (HTML5 canvas + vanilla JS)

Build a top-down GTA2-style driving game as a **static web page**, in this directory.
Two files only:

- `index.html` — a canvas page (canvas `960×640`), inline `<style>`, and
  `<script src="game.js"></script>`. No build step, no bundler, no external assets.
- `game.js` — all game code in **plain browser JavaScript**.

Hard tech constraints (the game is graded against these):

- **One `requestAnimationFrame` loop** named `gameLoop`: read input → update
  fixed-`dt` physics → render. No `setInterval`, no second loop.
- **No ES modules, no `class`, no `async`, no IIFE.** Top-level `let`/`const` only.
- Read `canvas.width`/`canvas.height` from JS; do not hardcode dimensions twice.
- All drawing is HTML5 canvas 2D (`getContext('2d')`). No DOM-element sprites, no WebGL.
- No network calls, no asset downloads — generate all shapes/colors in code.
- The page must load with **zero console errors** and the loop must run every frame.

Implement these milestones in order. Each is graded independently — a clean
M1–M5 beats a crashing M1–M8. Favor correctness and runnability over scope.

- **M1 — project shape + loop**: `index.html` canvas skeleton + `game.js` with the
  `gameLoop` requestAnimationFrame callback and `keydown`/`keyup` listeners that
  populate a `keys` object (use `event.code`, not `event.key`).
- **M2 — world map**: a tile grid stored as a multi-line string (alphabet e.g.
  `.`=road, `=`=sidewalk, `B`=building), 32px tiles, ~40×30 → larger than the
  viewport. Render by iterating the grid through a single `drawTile(char,x,y)`.
  Must show streets + buildings, not a blank canvas.
- **M3 — player car**: a `player = { x, y, angle, speed, ... }` object literal.
  Render with the `save → translate(x,y) → rotate(angle) → draw → restore`
  transform (body, roof, windshield, headlights, wheels). Spawn on a road tile.
- **M4 — driving + collision**: `dt`-based motion. Forward accelerates; reverse
  brakes then reverses; friction returns speed to 0. **Steering proportional to
  speed** (stationary car cannot turn). Building collision via center-vs-tile
  (`floor(x/32), floor(y/32)`) — revert position + small bounce on hit.
- **M5 — camera**: a `camera = {x,y}` holding the screen's top-left world coord;
  `ctx.translate(-camera.x,-camera.y)` once around the world render. Follow the
  player and clamp to map bounds. Map scrolls as the car drives.
- **M6 — pedestrians + run-over + score**: a `peds` array spawned on sidewalk
  tiles; they wander and panic near the moving car. Running one over (proximity +
  speed) kills it and adds to `score`.
- **M7 — HUD**: after the world pass, reset the transform
  (`ctx.setTransform(1,0,0,1,0,0)`) and draw screen-space HUD: `SCORE: <n>` and a
  health bar. HUD must not scroll with the camera.
- **M8 — police + wanted (stretch)**: a `wanted` level that rises on crimes; a
  `police` array of cop cars that spawn by wanted level and chase the player.

Constraints recap: pure HTML/CSS/JS, `index.html` + `game.js`, canvas 2D only,
one `requestAnimationFrame` loop, no modules/classes/async, zero console errors.
