from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent

def get_absolute_path(file_name: str) -> str:
    """
    Return absolute path to an asset in this assets package.
    """
    path = BASE_PATH / file_name
    if not path.exists():
        raise FileNotFoundError(f"Asset '{file_name}' not found in {path.parent}")

    return str(path.resolve())