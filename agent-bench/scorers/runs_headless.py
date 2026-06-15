"""`runs_headless` — boot the generated web game under headless chromium.

The bench builds an HTML5-canvas / vanilla-JS game (index.html + game.js). "Ran"
means: the page loads without console/page errors, a requestAnimationFrame loop
is ticking, and the canvas has drawn something non-blank. We load it under
Playwright headless chromium, let it run a short window, then read those signals.
"""

from __future__ import annotations

from pathlib import Path

from adapters.base import RunArtifacts

from scorers.base import Scores

RUN_WINDOW_MS = 3000
# A correct game fires `load` within a beat (assets are inline). A game that hangs
# here is blocking the main thread (e.g. an unbounded init loop) — fail it fast.
GOTO_TIMEOUT_MS = 10000
RAF_INIT_SCRIPT = (
    "window.__rafTicks=0; const _r=window.requestAnimationFrame; "
    "window.requestAnimationFrame=function(cb){window.__rafTicks++; return _r(cb);};"
)
RENDER_SCRIPT = """
() => {
  const cv = document.querySelector('canvas');
  if (!cv) return {ok: false, reason: 'no canvas'};
  const ctx = cv.getContext('2d');
  if (!ctx) return {ok: false, reason: 'no 2d context'};
  const w = cv.width, h = cv.height;
  if (!w || !h) return {ok: false, reason: 'zero-size canvas'};
  const data = ctx.getImageData(0, 0, w, h).data;
  const seen = new Set();
  for (let i = 0; i < data.length; i += 400 * 4) {
    seen.add(data[i] + ',' + data[i+1] + ',' + data[i+2] + ',' + data[i+3]);
    if (seen.size > 1) return {ok: true};
  }
  return {ok: false, reason: 'blank canvas'};
}
"""


class RunsHeadlessScorer:
    name = "runs_headless"

    def score(self, workspace: Path, artifacts: RunArtifacts) -> Scores:
        entry = self._find_entry(workspace)
        if entry is None:
            return {"runs_headless": False, "reason": "no html entrypoint"}

        # A scorer must never crash the runner: swallow any launch/Playwright error.
        try:
            return self._run(entry)
        except Exception as exc:
            return {"runs_headless": False, "reason": f"playwright error: {exc}"}

    @staticmethod
    def _find_entry(workspace: Path) -> Path | None:
        # Resolve to absolute: Path.as_uri() (used below) rejects relative paths.
        index = workspace / "index.html"
        if index.exists():
            return index.resolve()
        hit = next((p for p in workspace.rglob("*.html") if ".git" not in p.parts), None)
        return hit.resolve() if hit is not None else None

    @staticmethod
    def _run(entry: Path) -> Scores:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright

        console_errors: list[str] = []
        page_errors: list[str] = []

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.on(
                    "console",
                    lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
                )
                page.on("pageerror", lambda exc: page_errors.append(str(exc)))
                page.add_init_script(RAF_INIT_SCRIPT)

                try:
                    page.goto(entry.as_uri(), wait_until="load", timeout=GOTO_TIMEOUT_MS)
                except PlaywrightTimeoutError:
                    # `load` never fired -> the page blocks the main thread (a synchronous
                    # infinite loop). The game does not run; score it failed, fast.
                    return {
                        "runs_headless": False,
                        "loads_clean": False,
                        "loop_alive": False,
                        "renders": False,
                        "raf_ticks": 0,
                        "console_errors": len(console_errors) + len(page_errors),
                        "reason": "load timeout (page blocks main thread — likely an init loop)",
                    }
                page.wait_for_timeout(RUN_WINDOW_MS)

                error_count = len(console_errors) + len(page_errors)
                loads_clean = error_count == 0
                raf_ticks = int(page.evaluate("window.__rafTicks || 0"))
                loop_alive = raf_ticks > 0

                renders = False
                render_reason = ""
                # The render probe must not crash the scorer: treat any failure as blank.
                try:
                    result = page.evaluate(RENDER_SCRIPT)
                    renders = bool(result.get("ok"))
                    if not renders:
                        render_reason = str(result.get("reason", "blank canvas"))
                except Exception as exc:
                    render_reason = f"render probe failed: {exc}"
            finally:
                browser.close()

        overall = loads_clean and loop_alive and renders
        if overall:
            reason = "ok"
        elif not loads_clean:
            reason = "console errors"
        elif not loop_alive:
            reason = "no rAF"
        else:
            reason = render_reason or "blank canvas"

        return {
            "runs_headless": overall,
            "loads_clean": loads_clean,
            "loop_alive": loop_alive,
            "renders": renders,
            "raf_ticks": raf_ticks,
            "console_errors": error_count,
            "reason": reason,
        }
