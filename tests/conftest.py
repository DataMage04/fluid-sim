from pathlib import Path
import sys

# Make repository modules importable when pytest collects tests from tests/.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
