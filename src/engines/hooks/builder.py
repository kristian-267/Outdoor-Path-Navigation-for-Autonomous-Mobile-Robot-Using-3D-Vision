import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from utils.registry import Registry


HOOKS = Registry("hooks")


def build_hooks(cfg):
    hooks = []
    for hook_cfg in cfg:
        hooks.append(HOOKS.build(hook_cfg))
    return hooks
