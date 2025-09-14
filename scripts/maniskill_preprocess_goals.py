import torch
from pathlib import Path

from agents.encoders.clip_lang_encoder import LangClip 
from environments.simulation.maniskill.utils.instructions import ENV_INSTRUCTIONS

output_dir = Path("environments/simulation/maniskill/utils/preprocessed_embeddings")
output_dir.mkdir(exist_ok=True)

print("Loading language encoder...")
lang_encoder = LangClip(freeze_backbone=True, model_name="RN50")
lang_encoder.eval() 
print("Encoder loaded.")

for env_id, instruction in ENV_INSTRUCTIONS.items():
    print(f"Processing: {env_id}")
    with torch.no_grad():
        embedding_tensor = lang_encoder([instruction])
    
    output_path = output_dir / f"{env_id}.pt"
    torch.save(embedding_tensor, output_path)
    print(f"Saved embedding to {output_path}")

print("\nPreprocessing complete.")