from pathlib import Path

__all__ = ["CACHE_DIR", "ROOT_DIR"]

CACHE_DIR = Path.home() / ".cache" / "matcha"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ROOT_DIR = Path(__file__).parent.parent.parent
