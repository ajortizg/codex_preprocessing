import os
import re
from pathlib import Path


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
