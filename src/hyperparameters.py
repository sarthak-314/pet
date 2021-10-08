from IPython.core.magic import register_line_cell_magic
from IPython import get_ipython
import yaml
import os
from .core import heading

# Install omegaconf if not already available
try:
    from omegaconf import OmegaConf
except:
    print('Installing omeaconf')
    os.system('pip install -q omegaconf')
    from omegaconf import OmegaConf


# Hyperparameters Magic Command
@register_line_cell_magic
def hyperparameters(_, cell):
    'Magic command to write hyperparameters into a yaml file and load it with omegaconf'
    # Save hyperparameters in experiment.yaml
    with open('experiment.yaml', 'w') as f:
        f.write(cell)

    # Load the YAML file into the variable HP
    HP = OmegaConf.load('experiment.yaml')
    get_ipython().user_ns['HP'] = HP

