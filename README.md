# CSC311 Project

Three-class classifier (Soccer / Baseball / Basketball) trained on student survey responses.

## Setup

We use [uv](https://docs.astral.sh/uv/) to manage Python and dependencies.

1. Install uv (once per machine), then open a new terminal:

   ```bash
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create the environment (installs Python 3.12 and the locked dependencies into `.venv/`):

   ```bash
   uv sync
   ```

3. Run scripts with `uv run` from the project root (no need to activate the venv):

   ```bash
   uv run python src/some_script.py
   ```

In VS Code, pick `.venv` via **Python: Select Interpreter**.

## Adding packages

Use `uv add <package>` (not `pip install`) and commit the updated `pyproject.toml` and `uv.lock`. Others then run `uv sync`.

After pulling, run `uv sync` again if `uv.lock` changed.

## Layout

```
data/         raw data and train/test/validation splits
src/          development code: preprocessing, data Exploration, training, prediction
submission/   final files uploaded to MarkUs
report/       LaTeX report and figures
starter/      course-provided files, reference only
```
