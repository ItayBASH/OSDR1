from pathlib import Path
import os

PACKAGE_ROOT = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
DATA_DIR = PACKAGE_ROOT / "data"
CACHED_ANALYSES_DIR = PACKAGE_ROOT / ".cached_analyses"


# make directories if they don't exist:
CACHED_ANALYSES_DIR.mkdir(parents=False, exist_ok=True)
