from __future__ import annotations

import sys
from pathlib import Path

# Make `import tantra` work when pytest is run from a plain checkout.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
