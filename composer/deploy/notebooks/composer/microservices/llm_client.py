"""Run the shared model facade from the course's existing source entry point."""
from pathlib import Path
import runpy
import sys

# Native lab mounts deployment source beneath composer/deploy; repository callers
# resolve the same service relative to task1.
source = next((root for root in (
    Path(__file__).resolve().parents[1] / "deploy/services/model",
    Path(__file__).resolve().parents[3] / "services/model",
) if (root / "llm_client.py").is_file()), None)
if source is None:
    raise SystemExit("Use docker compose to build and run the model service")
sys.path.insert(0, str(source))
runpy.run_path(str(source / "llm_client.py"), run_name=__name__)
