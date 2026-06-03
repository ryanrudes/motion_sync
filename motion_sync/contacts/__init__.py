"""Project-specific contact type definitions."""

from motion_sync.contacts.binary import BinaryContact, BinaryContactData, BinarySubjectTrack
from motion_sync.contacts.categorical import (
    CategoricalContact,
    CategoricalContactData,
    CategoricalSubjectTrack,
)
from motion_sync.contacts.foot_support import (
    FootSupport,
    FootSupportData,
    FootSupportState,
    FootSupportTrack,
    foot_support_config,
    layer_from_foot_classification,
)
from motion_sync.contacts.shoe_board_grip import ShoeBoardGrip, ShoeBoardGripData

__all__ = [
    "BinaryContact",
    "BinaryContactData",
    "BinarySubjectTrack",
    "CategoricalContact",
    "CategoricalContactData",
    "CategoricalSubjectTrack",
    "FootSupport",
    "FootSupportData",
    "FootSupportState",
    "FootSupportTrack",
    "ShoeBoardGrip",
    "ShoeBoardGripData",
    "foot_support_config",
    "layer_from_foot_classification",
]
