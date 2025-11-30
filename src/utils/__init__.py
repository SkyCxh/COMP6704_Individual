"""
Utility modules for latent CoT training.

This __init__.py re-exports everything from the original src.utils module
plus the new LatentGradientTracker.
"""

# Import all from the original utils.py file (at src level)
# We need to temporarily import from parent to avoid circular imports
import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import from src.utils (the file, not the package)
import importlib.util
_utils_file_path = os.path.join(_parent_dir, 'utils.py')
_spec = importlib.util.spec_from_file_location("_src_utils", _utils_file_path)
_utils_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils_module)

# Re-export all functions from utils.py
set_seed = _utils_module.set_seed
get_rank = _utils_module.get_rank
is_main_process = _utils_module.is_main_process
setup_distributed = _utils_module.setup_distributed
cleanup_distributed = _utils_module.cleanup_distributed
save_checkpoint = _utils_module.save_checkpoint
load_checkpoint = _utils_module.load_checkpoint
extract_answer = _utils_module.extract_answer
normalize_answer = _utils_module.normalize_answer
compute_accuracy = _utils_module.compute_accuracy
AverageMeter = _utils_module.AverageMeter
Logger = _utils_module.Logger

# Import from gradient_tracker module
from .gradient_tracker import LatentGradientTracker

__all__ = [
    # From original utils.py
    'set_seed',
    'get_rank',
    'is_main_process',
    'setup_distributed',
    'cleanup_distributed',
    'save_checkpoint',
    'load_checkpoint',
    'extract_answer',
    'normalize_answer',
    'compute_accuracy',
    'AverageMeter',
    'Logger',
    # From gradient_tracker
    'LatentGradientTracker',
]

