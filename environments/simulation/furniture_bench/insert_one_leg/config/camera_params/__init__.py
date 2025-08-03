from pathlib import Path
import yaml


def load(name: str):
    """Load YAML from the camera_params package by name (with or without .yaml).
    This way we can just load the parent folder as a module and extract the file using the name specified by the user anywhere, instead of hard-coding paths.
    """
    filename = f"{name}.yaml" if not name.endswith(".yaml") else name
    base_path = Path(__file__).parent
    file_path = base_path / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Camera parameter file '{file_path}' not found")
    with file_path.open("r") as f:
        return yaml.safe_load(f)
