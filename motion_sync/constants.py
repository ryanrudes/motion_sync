"""Default Vicon body names and config path for skate trials."""

from pathlib import Path

SKATEBOARD_VICON_NAME = "Skateboard"
"""Skateboard rigid-body name in Vicon exports."""

RIGHT_SHOE_VICON_NAME = "Right_Shoe"
"""Right shoe rigid-body name in Vicon exports."""

LEFT_SHOE_VICON_NAME = "Left_Shoe"
"""Left shoe rigid-body name in Vicon exports."""

DEFAULT_CONFIG_PATH = Path("configs/motion_sync.yaml")
"""Default YAML path for :func:`~motion_sync.config.load_config`."""
