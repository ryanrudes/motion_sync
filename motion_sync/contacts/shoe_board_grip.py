"""Binary shoe-on-board grip derived from foot-support classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np

from motion_sync.contact_layer import ContactLayer
from motion_sync.contacts.binary import BinaryContact, BinaryContactData
from motion_sync.contacts.foot_support import FootSupport, FootSupportState

BodyT = TypeVar("BodyT", bound=StrEnum)


@dataclass(frozen=True)
class ShoeBoardGrip(BinaryContact["ShoeBoardGripData"], Generic[BodyT]):
    """True when a shoe is classified as on the skateboard (not air or ground only)."""

    layer_id: ClassVar[str] = "shoe_board_grip"

    left: BodyT
    right: BodyT
    foot_support: FootSupport[BodyT]

    @property
    def _subjects(self) -> tuple[BodyT, BodyT]:
        return (self.left, self.right)

    def detect(self, clip: Any, config: Any | None = None) -> ContactLayer:
        if clip.has_contact(self.foot_support):
            fs_layer = clip.contact_layer(self.foot_support.layer_id)
            if not clip.contact_is_fresh(self.foot_support):
                fs_layer = self.foot_support.detect(clip, config)
        else:
            fs_layer = self.foot_support.detect(clip, config)
        data = self.foot_support.read(clip, fs_layer)
        left_on_board = data.track(self.left).states == int(FootSupportState.SKATEBOARD)
        right_on_board = data.track(self.right).states == int(FootSupportState.SKATEBOARD)
        return self.build_layer(
            subjects=(self.left.value, self.right.value),
            mask=np.column_stack([left_on_board, right_on_board]),
            metadata={"derived_from": self.foot_support.layer_id},
        )

    def read(self, clip: Any, layer: ContactLayer) -> ShoeBoardGripData[BodyT]:
        self._validate_layer(layer)
        return ShoeBoardGripData(
            layer,
            contact=self,
            time_s=np.asarray(clip.time_s, dtype=np.float64),
        )


class ShoeBoardGripData(BinaryContactData[BodyT], Generic[BodyT]):
    """Read API for an attached shoe-board grip layer."""

    def __init__(
        self,
        layer: ContactLayer,
        *,
        contact: ShoeBoardGrip[BodyT],
        time_s: np.ndarray,
    ) -> None:
        super().__init__(layer, subjects=contact._subjects, time_s=time_s)
        self._contact = contact
