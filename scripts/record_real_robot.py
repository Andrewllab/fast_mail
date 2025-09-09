import hydra
import rootutils
from hydra.utils import instantiate
from omegaconf import DictConfig

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

from environments.real_robot.teleop.data_collector import DataCollectionManager


@hydra.main(
    version_base=None, config_path="../configs", config_name="record_real_robot"
)
def main(cfg: DictConfig):

    data_collection_manager: DataCollectionManager = instantiate(
        cfg.data_collection_manager,
        _target_=DataCollectionManager,
    )

    data_collection_manager.start_key_listener()


if __name__ == "__main__":
    main()
