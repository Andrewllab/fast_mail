from typing import List

import hydra
from hydra.utils import instantiate
import rootutils
from omegaconf import DictConfig

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

from environments.real_robot.teleop.data_collector import DataCollectionManager
from environments.real_robot.teleop.teleoperation_base import TeleoperationPair
from environments.real_robot.hardware.base_camera import BaseCamera

@hydra.main(version_base=None, config_path="../configs", config_name="record_real_robot")
def main(cfg: DictConfig):

    teleoperation_pair: TeleoperationPair = instantiate(cfg.teleoperation_pair)
    
    # Check if continuous devices are defined in the config
    if 'cameras' in cfg:
        cameras: List[BaseCamera] = [instantiate(d) for d in cfg.cameras.values()]
    else:
        cameras = []
        
    data_collection_manager: DataCollectionManager = instantiate(
        cfg.data_collection_manager,
        _target_=DataCollectionManager,
        teleoperation_pair=teleoperation_pair,
        cameras=cameras
    )

    data_collection_manager.start_key_listener()

if __name__ == "__main__":
    main()
