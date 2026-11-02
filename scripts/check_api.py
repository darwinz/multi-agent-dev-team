"""CLI wrapper: run the Postman fixture against a generated FastAPI app."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api_check import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
