import logging
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from codex_preprocessing.utils import configure_tensorflow_gpus, task_wrapper

# CONFIG_DIR = str(Path(__file__).parent.parent.parent / "config")

log = logging.getLogger(__name__)


@task_wrapper
def preprocess(cfg: DictConfig):
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    # configure_tensorflow_gpus(True)
    log.info(f"{OmegaConf.to_yaml(cfg, resolve=True)}")

    data = hydra.utils.instantiate(cfg.data)
    pipeline = hydra.utils.instantiate(cfg.pipeline, data=data, out_dir=out_dir)
    pipeline.run()

    return None


@hydra.main(config_path="./config", config_name="preprocess", version_base=None)
def main(cfg: DictConfig):
    log.info(f"{torch.cuda.is_available()} GPUs detected: {torch.cuda.device_count()}")
    preprocess(cfg)


if __name__ == "__main__":
    main()
