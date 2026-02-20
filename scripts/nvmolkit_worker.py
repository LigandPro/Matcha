#!/usr/bin/env python3
"""Compatibility wrapper for the packaged worker entrypoint."""

import sys


def _main() -> int:
    try:
        from matcha_nvmolkit_worker.cli import main
    except Exception as exc:
        print(
            "Failed to import matcha_nvmolkit_worker package. "
            "Run: uv run python scripts/setup_nvmolkit_worker.py",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1
    return int(main())


if __name__ == "__main__":
    raise SystemExit(_main())
