import os
from pathlib import Path


def resolve_path(str_path: os.PathLike) -> Path:
    # resolve any environment variables in the first path element
    elements = str(str_path).split("/")
    # if first character of first element is "$"
    if (root := elements[0]).startswith("$"):
        elements[0] = os.environ[root[1:]]
    path = "/".join(elements)  # not osp.join because we used str.split above

    # resolve ~ to the user's home directory and resolve symlinks
    path = Path(path).expanduser().resolve()
    return path


def iglob_follow_symlinks(root: Path, pattern: str, include_dirs: bool = False):
    """Yield Paths under `root` that match `pattern`, following directory symlinks.
    Works like Path.glob with '**' but follows links (Python 3.10 compatible).

    `pattern` is matched from the repo root (like Path.match), e.g. '**/*.py'.
    Set include_dirs=True to also yield matching directories.
    """
    root = Path(root).resolve()
    seen_dirs = set()  # (st_dev, st_ino) keys to avoid cycles

    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        # De-duplicate / break cycles
        try:
            st = os.stat(dirpath)
        except FileNotFoundError:
            # directory disappeared between os.walk and stat
            continue
        key = (st.st_dev, st.st_ino)
        if key in seen_dirs:
            # Already visited this real directory via a different symlink path
            dirnames[:] = []  # don't descend further
            continue
        seen_dirs.add(key)

        # Prune children that would lead to already-seen directories
        keep = []
        for d in dirnames:
            p = os.path.join(dirpath, d)
            try:
                st_child = os.stat(p)  # follows symlinks
            except FileNotFoundError:
                continue
            if (st_child.st_dev, st_child.st_ino) not in seen_dirs:
                keep.append(d)
        dirnames[:] = keep

        dir = Path(dirpath)  # .relative_to(root)

        if include_dirs:
            for d in dirnames:
                p = dir / d
                if p.match(pattern):
                    yield p

        for f in filenames:
            p = dir / f
            if p.match(pattern):
                yield p
