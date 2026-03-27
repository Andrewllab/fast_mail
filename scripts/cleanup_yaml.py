from typing import Any, Dict

import yaml

IN_FILE = "annotations_stack.yaml"
OUT_FILE = "annotations_stack.yaml"

CAMERAS = ["left_cam", "right_cam"]
OBJECT_TYPES = ["tool object", "target object"]


def clean_empty_fields(data):
    if isinstance(data, dict):
        cleaned_data = {}
        for k, v in data.items():
            cleaned_v = clean_empty_fields(v)
            if cleaned_v is not None and cleaned_v != "":
                cleaned_data[k] = cleaned_v

        if not cleaned_data:
            return None

        return cleaned_data

    elif isinstance(data, list):
        cleaned_data = [
            clean_empty_fields(item) for item in data if item is not None and item != ""
        ]
        if not cleaned_data:
            return None
        return cleaned_data
    else:
        return data


def convert_structure(data: Dict[str, Any]) -> Dict[str, Any]:
    input_files = data.get("files", {})
    output = {"files": {}}

    for filename, file_data in input_files.items():
        out_file = {}

        timesteps = file_data.get("timesteps", {})
        if not isinstance(timesteps, dict):
            timesteps = {}

        for timestep_str, timestep_data in timesteps.items():
            try:
                timestep = int(timestep_str)
            except (TypeError, ValueError):
                timestep = timestep_str

            if not isinstance(timestep_data, dict):
                continue

            for camera_name, camera_data in timestep_data.items():
                if camera_name not in CAMERAS:
                    continue
                if not isinstance(camera_data, dict):
                    continue

                out_file.setdefault(camera_name, {})

                for object_type, points in camera_data.items():
                    if object_type not in OBJECT_TYPES:
                        continue
                    if not isinstance(points, list):
                        continue

                    object_type = object_type.replace(" ", "_")
                    out_file[camera_name].setdefault(object_type, [])

                    for pt in points:
                        if not isinstance(pt, dict):
                            continue
                        if "x" not in pt or "y" not in pt:
                            continue

                        out_file[camera_name][object_type].append(
                            {
                                "x": pt["x"],
                                "y": pt["y"],
                                "timestep": timestep,
                            }
                        )

        output["files"][filename] = out_file

    return output


if __name__ == "__main__":
    with open(IN_FILE, "r") as f:
        data = yaml.safe_load(f)
        new_data = clean_empty_fields(data)
    new_data["files"] = {
        k: v for k, v in new_data.get("files", {}).items() if "2026" in k
    }

    new_data = convert_structure(new_data)

    with open(OUT_FILE, "w") as f:
        yaml.dump(new_data, f, default_flow_style=False)
