"""Write the committed JSON Schemas for the versioned research API."""

from __future__ import annotations

import json
from pathlib import Path

from scripture_fidelity.api import api_schema_documents


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"


def main() -> None:
    SCHEMA_DIR.mkdir(exist_ok=True)
    for name, schema in api_schema_documents().items():
        (SCHEMA_DIR / name).write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
