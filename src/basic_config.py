"""
This is the demo code that uses hydra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from config import MainConfig

cs = ConfigStore.instance()
cs.store(name="mainconfig", node=MainConfig)


@hydra.main(config_path="../config", config_name="main", version_base=None)
def process_data(config: MainConfig):
    """Function to process the data"""

    print(config, type(config))
    print(config.main_params.fpr_max, type(config.main_params.fpr_max))
    print(config.model_params["199"], type(config.model_params["199"]))
    print(type(config.main_params))
    print(isinstance(config, MainConfig))
    config_dict = OmegaConf.to_container(config, throw_on_missing=False, resolve=True)
    cfg_obj = OmegaConf.to_object(config)
    print(cfg_obj, type(cfg_obj))
    print(isinstance(cfg_obj, MainConfig))
    print(type((cfg_obj["local_paths"])))
    print(isinstance(config_dict, MainConfig))
    print(type((config_dict["local_paths"])))
    print(config_dict)
    conf = OmegaConf.structured(config)
    print(conf, type(conf))
    print(isinstance(conf, MainConfig))


if __name__ == "__main__":
    process_data()
