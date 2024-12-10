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


@hydra.main(config_path="configs", config_name="libero_horeka_config.yaml", version_base="1.3")
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

    # load vqvae before training the agent: add path to the config file
    # train the agent
    agent = hydra.utils.instantiate(cfg.agents)

    ###################################################
    # load the mask-pretrained model
    ###################################################
    pretrain_dir = sim_framework_path("pretrain_weights")
    agent.load_pretrained_model(pretrain_dir, sv_name=f"batch{cfg.mask_batch_size}_{cfg.task_suite}_{cfg.agent_name}.pth")

    # simulate the model
    env_sim = hydra.utils.instantiate(cfg.simulation)

    sv_dir = sim_framework_path("pretrain_weights")

    for num_epoch in tqdm(range(agent.epoch)):

        agent.train_agent()

        if num_epoch in [49, 69, 89, 99]:

            env_sim.test_agent(agent, epoch=num_epoch)

            # Save the model checkpoint
            # torch.save(agent.model.state_dict(), sv_dir + f'/{num_epoch}.pth')

    log.info("Training done")
    log.info("state_dict saved in {}".format(agent.working_dir))
    wandb.finish()


if __name__ == "__main__":
    main()
