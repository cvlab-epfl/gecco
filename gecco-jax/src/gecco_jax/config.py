import os
import re
from importlib.machinery import SourceFileLoader
from types import ModuleType
from typing import Tuple, Union

CHECKPOINT_SAVE_TEMPLATE = 'checkpoint-step-{}'
CHECKPOINT_SAVE_RE = re.compile(r'checkpoint-step-(\d+)')

def load_config(config_path: str) -> ModuleType:
    if not config_path.endswith('.py'):
        raise ValueError(f'{config_path=} does not end in .py')

    return SourceFileLoader('config', config_path).load_module()

def latest_checkpoint_legacy(
    config_path: str,
    ema: bool = False,
    return_step_number: bool = False,
) -> Union[str, Tuple[str, int]]:
    if ema:
        checkpoint_re = re.compile(r'ema_(\d+)\.eqx')
    else:
        checkpoint_re = re.compile(r'(\d+)\.eqx')

    path = None
    highest_id = 0

    for file in os.listdir(config_path):
        if (m := checkpoint_re.match(file)) is not None:
            save_id = int(m.group(1))
            if save_id > highest_id:
                highest_id = save_id
                path = os.path.join(config_path, file)
    
    if path is None:
        raise IOError(f'No checkpoints found in {config_path=}.')
    
    if return_step_number:
        return path, highest_id
    else:
        return path

def latest_checkpoint(
    config_path: str,
    return_step_number: bool = False,
) -> Union[str, Tuple[str, int]]:
    path = None
    highest_id = 0

    for file in os.listdir(config_path):
        if (m := CHECKPOINT_SAVE_RE.match(file)) is not None:
            save_id = int(m.group(1))
            if save_id > highest_id:
                highest_id = save_id
                path = os.path.join(config_path, file)
    
    if path is None:
        raise IOError(f'No checkpoints found in {config_path=}.')
    
    if return_step_number:
        return path, highest_id
    else:
        return path