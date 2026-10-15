from pathlib import Path
import sys

import matplotlib

# Tests must not depend on a desktop GUI/Tk installation. This keeps plotting
# smoke tests portable while leaving the application's normal backend unchanged.
matplotlib.use("Agg", force=True)

# Make repository modules importable when pytest collects tests from tests/.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
