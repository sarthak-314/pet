"""
Tensorflow Startup Script
- Load TPU / GPU
- Mixed precision, XLA Accelerate
"""
import tensorflow as tf
import os

# Install tensorflow addons & tensorflow hub if not already available
try:
    import tensorflow_addons as tfa
except:
    print('Installing tensorflow_addons')
    os.system('pip install -q tensorflow_addons')
    import tensorflow_addons as tfa
try:
    import tensorflow_hub as hub
except:
    print('Installing tensorflow hub')
    os.system('pip install -q tensorflow_hub')
    import tensorflow_hub as hub


from ..core import HARDWARE, ENV

# Module Imports for Notebook
from .factory import (
    lr_scheduler_factory, optimizer_factory, callbacks_factory,
    get_wandb_callback, build_hidden_layer,
)
from .augmentations import augmentations_factory, augmentations_config


def _enable_mixed_precision():
    """
    - Model runs faster and uses less memory
    - More than 60% performance improvement on TPU
    """
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

def _enable_xla_acceleration():
    """
    https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html
    - Don't use variable sizes with TPU. Compile time adds up
    - Uses extra memory
    - Don't use for short scripts
    """
    tf.config.optimizer.set_jit(True)


def tf_accelerator(bfloat16=True, jit_compile=True):
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")

    if HARDWARE is 'CPU':
        print('CPU detected. Skipping mixed precision and jit compilation')
        return strategy

    if bfloat16:
        _enable_mixed_precision()
        print('Mixed precision enabled')
    if jit_compile:
        _enable_xla_acceleration()

    return strategy

def adjust_batch_size_tpu(batch_size):
    if HARDWARE == 'CPU':
        print('Using a batch size of 2 for CPU')
        return 2
    if ENV == 'Colab' and HARDWARE == 'TPU':
        # Colab TPU have less memory than Kaggle TPUs
        print('Halfing batch size for Colab TPU')
    print('using batch size: ', batch_size)
    return batch_size


def get_gcs_path(dataset_name):
    from kaggle_datasets import KaggleDatasets
    gcs_path = KaggleDatasets().get_gcs_path(dataset_name)
    print(f'GCS path for {dataset_name}: ', gcs_path)
    return gcs_path