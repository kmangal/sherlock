from pathlib import Path


class BackendRootNotFound(Exception):
    pass


def get_backend_root_dir() -> Path:
    """Look for the root pyproject.toml"""
    current = Path(__file__).parent

    # Walk up until we hit the root
    while not (current.is_dir() and current.name == "sherlock"):
        candidate = current / "pyproject.toml"
        if candidate.exists():
            return current

        # Check if we've reached the root
        parent = current.parent
        if parent == current:
            raise BackendRootNotFound

        current = parent

    raise BackendRootNotFound
