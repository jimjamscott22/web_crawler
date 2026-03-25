## Learned User Preferences

- Prefer `uv run` for this project without activating a separate `venv` that does not match the environment `uv` uses (typically `.venv` from `uv sync`).
- If the integrated terminal auto-activates Python on open, point the workspace interpreter at this project’s `.venv` so `VIRTUAL_ENV` aligns with `uv run` and avoids mismatch warnings.
- Optionally disable `python.terminal.activateEnvironment` in workspace settings when a non-activated shell or manual activation is preferred.

## Learned Workspace Facts

- Dependencies are managed with uv (`pyproject.toml`, `uv.lock`); `uv sync` creates `.venv` and installs from the lockfile.
- Default crawl exports and `crawler.log` are written under `scrape_results/` next to `crawler.py`, not tied to the shell’s current working directory; absolute export paths are respected as given.
