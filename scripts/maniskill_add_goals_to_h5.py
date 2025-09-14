import sys
import os
import argparse
import h5py
import numpy as np
import torch
from pathlib import Path

from environments.simulation.maniskill.utils.instructions import ENV_INSTRUCTIONS

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

parser = argparse.ArgumentParser(description="Inject a pre-processed goal embedding into an HDF5 file.")
parser.add_argument("--filepath", type=str, required=True, help="Path to the .h5 trajectory file.")
parser.add_argument("--env-id", type=str, required=True, help="Environment ID (e.g., 'StackCube-v1') to find the corresponding .pt file.")
parser.add_argument("--embeddings-dir", type=str, default="environments/simulation/maniskill/utils/preprocessed_embeddings", help="Directory containing pre-processed .pt files.")
args = parser.parse_args()

instruction_text = ENV_INSTRUCTIONS[args.env_id]

embedding_path = Path(args.embeddings_dir) / f"{args.env_id}.pt"

if not embedding_path.exists():
    raise FileNotFoundError(
        f"Could not find pre-processed embedding for '{args.env_id}' at {embedding_path}. "
        "Please ensure the file exists and the env-id is correct."
    )

print(f"Loading pre-processed embedding from: {embedding_path}")
embedding_tensor = torch.load(embedding_path)
embedding_np = embedding_tensor.squeeze(0).cpu().numpy()

print(f"Injecting embedding for '{args.env_id}' into {args.filepath}")

with h5py.File(args.filepath, "r+") as hf:
    for traj_name in hf.keys():
        traj_group = hf[traj_name]
        if not isinstance(traj_group, h5py.Group):
            continue

        goal_group = traj_group.require_group("goal")

        if "text" in goal_group:
            print(f"'goal/text' already exists in {traj_group.name}. Skipping text.")
        else:
            goal_group.create_dataset("text", data=np.string_(instruction_text))
            print(f"Successfully added 'goal/text' to {traj_group.name}.")

        if "embedding" in goal_group:
            print(f"'goal/embedding' already exists in {traj_group.name}. Skipping embedding.")
        else:
            goal_group.create_dataset("preprocessed_embedding", data=embedding_np)
            print(f"Successfully added 'goal/embedding' to {traj_group.name}.")

print("Script finished.")