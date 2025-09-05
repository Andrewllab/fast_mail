import logging

import hydra
import rootutils
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)
from environments.datamodule import TrajectoryDataModule
from loggers.wandb import update_wandb_config
from utils.conf import log_slurm_job_id, setup_resolvers
from utils.instantiators import instantiate_datamodule, instantiate_loggers
from utils.logging import configure_logging, log_exception_and_finish_wandb
from utils.torch_conf import configure_torch

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
@log_exception_and_finish_wandb
def prepreprocess(cfg: DictConfig) -> None:
    # resolve the entire config to catch any errors early
    OmegaConf.resolve(cfg)
    log_slurm_job_id(cfg)

    configure_logging(cfg.python_logging)

    # init wandb first so we can log any info or errors from instantiating dataset and model
    log.debug("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))
    update_wandb_config(cfg)

    # configure torch, e.g. set_float32_matmul_precision
    configure_torch(cfg.get("torch"))

    # TODO: filter out any preprocess steps from the dataset in case the user
    # forgot to do this

    # instantiate dataset
    datamodule: TrajectoryDataModule = instantiate_datamodule(cfg.data)

    # manually run prepare data and setup so we can use dataset specs for model creation
    log.debug("Loading training data...")
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    log.info("Prepreprocessing completed.")

    datamodule.close()


if __name__ == "__main__":
    setup_resolvers()
    prepreprocess()
