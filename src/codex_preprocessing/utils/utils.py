import logging
import os
import re
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict

from omegaconf import DictConfig

log = logging.getLogger(__name__)


def ensure_path(path: str | Path) -> Path:
    if isinstance(path, str):
        path = Path(path)
    return path


def configure_tensorflow_gpus(use_gpu):

    import tensorflow as tf

    if use_gpu is False:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    else:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if len(gpus) == 1:
            n = re.match(r"/physical_device:GPU:(?P<n>\d+)", gpus[0].name).groupdict()["n"]
            # print(f"------------------- running on gpu {n} ---------------------")
        elif len(gpus) > 1:
            n = re.match(r"/physical_device:GPU:(?P<n>\d+)", gpus[1].name).groupdict()["n"]
            # print(f"------------------- running on gpu {n} ---------------------")
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Dict[str, Any]:
        # execute the task
        try:
            object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return object_dict

    return wrap
