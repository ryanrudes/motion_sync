"""Register contact types on a clip (parallel to :class:`~motion_sync.mocap_schema.MocapSchema`)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, ClassVar, Generic, TypeVar

from motion_sync.contact_layer import ContactLayer

StateT = TypeVar("StateT", bound=IntEnum)
ViewT = TypeVar("ViewT")


def state_label_strings(state_enum: type[IntEnum]) -> tuple[str, ...]:
    """Lowercase enum member names for on-disk ``ContactLayer.labels``."""
    return tuple(member.name.lower() for member in state_enum)


class ContactType(ABC, Generic[StateT, ViewT]):
    """Define a contact kind once, then pass this object to :meth:`SyncClip.register_contacts`.

    Subclass for each detector (foot support, hand grip, …). Set :attr:`layer_id` and
    :attr:`State` (an :class:`IntEnum` of per-frame categories). Implement :meth:`detect`
    and :meth:`read`.
    """

    layer_id: ClassVar[str]

    State: ClassVar[type[IntEnum]]

    @classmethod
    def labels_on_disk(cls) -> tuple[str, ...]:
        return state_label_strings(cls.State)

    @abstractmethod
    def detect(self, clip: Any, config: Any | None = None) -> ContactLayer:
        """Run detector; return layer aligned to ``clip.time_s`` (does not attach)."""

    @abstractmethod
    def read(self, clip: Any, layer: ContactLayer) -> ViewT:
        """Typed accessor over an attached layer."""


@dataclass(frozen=True)
class ContactSchema:
    """Bundle of contact types for a project (e.g. skate)."""

    types: tuple[ContactType[Any, Any], ...]

    def __iter__(self):
        return iter(self.types)


def merge_registered_contacts(
    existing: dict[str, ContactType[Any, Any]] | None,
    *contact_types: ContactType[Any, Any],
) -> dict[str, ContactType[Any, Any]]:
    merged = dict(existing or {})
    for contact in contact_types:
        if contact.layer_id in merged and merged[contact.layer_id] is not contact:
            raise ValueError(f"duplicate contact type layer_id {contact.layer_id!r}")
        merged[contact.layer_id] = contact
    return merged
