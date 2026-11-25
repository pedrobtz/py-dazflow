"""The example project (examples/loss-rate) and a snapshot of a folder's files."""

from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "loss-rate"


def tree(root: Path) -> dict[str, bytes]:
    """Every file under `root` by its relative path, with its bytes."""
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
