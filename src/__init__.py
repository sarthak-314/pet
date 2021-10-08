"""
Startup script to run in the beggining of every Jupyter Notebook for the competition
- Import common libraries
- Jupyter Notebook Setup: autoreload, display all, add to sys.path
- Import common functions, classes & constants
- Import competition specific functions / constants
"""

# Commonly Used Libraries
from functools import partial
from termcolor import colored
from tqdm.auto import tqdm
from pathlib import Path
from time import time
import pandas as pd
import numpy as np
import random
import pickle
import sys
import os
import re

# Uncommonly Used Libraries
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from IPython.display import display, Markdown
from dataclasses import dataclass, asdict
from distutils.dir_util import copy_tree
from IPython.display import clear_output
from collections import defaultdict
import matplotlib.pyplot as plt
from IPython import get_ipython
from PIL import Image
import subprocess
import warnings
import shutil
import math
import glob
import json
import cv2
import gc

# Package Imports
from .hyperparameters import hyperparameters
from .integrations import wandb_init
from .core import (
    ENV, HARDWARE, IS_ONLINE, KAGGLE_INPUT_DIR, WORKING_DIR, TMP_DIR,
    red, blue, green, yellow,
)


def _setup_jupyter_notebook():
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = 'all'
    ipython = get_ipython()
    try:
        ipython.magic('matplotlib inline')
        ipython.magic('load_ext autoreload')
        ipython.magic('autoreload 2')
    except:
        print('could not load ipython magic extensions')
_setup_jupyter_notebook()


def _ignore_deprecation_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
_ignore_deprecation_warnings()

# Startup Notebook Functions
REPO_PATH = 'https://github.com/sarthak-314/pet'
def sync():
    'Sync Notebook with VS Code'
    os.chdir(WORKING_DIR/'pet')
    subprocess.run(['git', 'pull'])
    sys.path.append(str(WORKING_DIR/'pet'))
    os.chdir(WORKING_DIR)

# Mount Drive in Colab
def _colab_mount_drive():
    from google.colab import drive
    drive.mount('/content/drive')
if ENV == 'Colab':
    _colab_mount_drive()



# Competition Specific Constants & Functions
COMP_NAME = 'petfinder-pawpularity-score'
DRIVE_DIR = Path('/content/drive/MyDrive/Pet')
DF_DIR = {
    'Kaggle': KAGGLE_INPUT_DIR/'pet-dataframes',
    'Colab': DRIVE_DIR/'Dataframes',
    'Surface Pro': WORKING_DIR/'data',
}[ENV]