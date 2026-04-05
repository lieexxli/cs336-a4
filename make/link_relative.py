import os
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: link_relative.py <target> <link_path>")

    target = Path(sys.argv[1]).resolve()
    link_path = Path(sys.argv[2])
    link_path.parent.mkdir(parents=True, exist_ok=True)

    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()

    relative_target = os.path.relpath(target, start=link_path.parent.resolve())
    link_path.symlink_to(relative_target)
    print(f"{link_path} -> {relative_target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
