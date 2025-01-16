import logging
import random
import os
import hydra
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
import torch

from agents.utils.sim_path import sim_framework_path

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path="configs", config_name="robocasa_pc_config.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:

    set_seed_everywhere(cfg.seed)

    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.group,
        # mode="disabled",
        config=wandb.config
    )

    # train the agent
    agent = hydra.utils.instantiate(cfg.agents)
    trainer = hydra.utils.instantiate(cfg.trainers)

    agent.get_params()
    # trainer.main(agent)

    agent.set_scaler(trainer.scaler)
    agent.load_pretrained_model(f'/hkfs/work/workspace/scratch/ll6323-david_dataset_2/weights/rgb_ada_false/obs_encoders=pretrained_resnet_film,agents=beso_agent,group=beso_encoder_decoder,seed={cfg.seed},xlstm_decoder_in_context_cond=True,xlstm_decoder_vocab_size=11',
                                sv_name='last_model')

    # simulate the model
    env_sim = hydra.utils.instantiate(cfg.simulation)
    # env_sim.get_task_embs(trainer.trainset.tasks)
    #
    # sv_dir = sim_framework_path("pretrain_weights")
    #
    env_sim.test_agent(agent)

    log.info("Training done")
    log.info("state_dict saved in {}".format(agent.working_dir))
    wandb.finish()


if __name__ == "__main__":
    main()
