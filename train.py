import logging

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict

from utils.conf import pop_names, setup_resolvers
from utils.logging import configure_logging
from utils.seeding import get_rng, manual_seed

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging)

    with open_dict(cfg):
        notes = cfg.wandb.pop("notes", None)

    run = wandb.init(
        project="3d-sim2real",
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # type: ignore
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        save_code=True,  # save script used to start training, git commit, and patch
        reinit=True,  # required for hydra sweeps with default launcher
        tags=cfg.wandb.get("tags"),
        notes=notes,
    )

    # recursively pop any "name" fields in config dictionary
    # we want these to be saved to WandB but we don't want them for instantiation
    cfg = pop_names(cfg)

    # BALAZS: do we need this?
    torch.cuda.empty_cache()

    # seeding
    rng = get_rng(cfg)
    manual_seed(rng)

    # instantiate dataset
    dataset = hydra.utils.instantiate(cfg.dataset.dataset)

    # instantiate agent
    agent = hydra.utils.instantiate(cfg.agent, dataset=dataset)

    device = torch.device(cfg.device)
    agent = agent.to(device)

    log.warning("Exiting after instantiating agent since script is not finished yet.")
    return

    trainer = hydra.utils.instantiate(
        cfg.trainers, trainset=dataset, dataloader_cfg=cfg.dataset.dataloader
    )

    agent.get_params()
    trainer.main(agent)

    # # simulate the model
    env_sim = hydra.utils.instantiate(cfg.simulation)
    env_sim.get_task_embs(trainer.trainset.tasks)
    # #
    # # sv_dir = sim_framework_path("pretrain_weights")
    # #
    # env_sim.test_agent(agent, cfg.agents)
    env_sim.test_agent(agent, cfg.agents, epoch=cfg.epoch)

    log.info("Training done")
    log.info("state_dict saved in {}".format(agent.working_dir))

    run.finish()


if __name__ == "__main__":
    setup_resolvers()
    main()
