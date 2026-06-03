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
    """Build on-disk categorical labels from an :class:`IntEnum`.

    Args:
        state_enum: Per-frame state enum for a contact detector.

    Returns:
        Lowercase member names, in enum definition order.
    """
    return tuple(member.name.lower() for member in state_enum)


class ContactType(ABC, Generic[StateT, ViewT]):
    """Define a contact kind once, then register it on :class:`~motion_sync.synced_dataset.SyncClip`.

    Subclass for each detector (foot support, shoe-board grip, …). Set :attr:`layer_id` and
    :attr:`State` (an :class:`IntEnum` of per-frame categories). Implement :meth:`detect`
    and :meth:`read`.

    Attributes:
        layer_id (str): Stable storage key (``[a-z][a-z0-9_]*``); class variable on subclasses.
        State (type[IntEnum]): Integer codes written to categorical layers; class variable.
    """

    layer_id: ClassVar[str]

    State: ClassVar[type[IntEnum]]

    @classmethod
    def labels_on_disk(cls) -> tuple[str, ...]:
        """Return labels stored in :class:`~motion_sync.contact_layer.ContactLayer.labels`.

        Returns:
            Lowercase names derived from :attr:`State`.
        """
        return state_label_strings(cls.State)

    @abstractmethod
    def detect(self, clip: Any, config: Any | None = None) -> ContactLayer:
        """Run the detector and build a layer aligned to ``clip.time_s``.

        Does not attach the layer; use :meth:`~motion_sync.synced_dataset.SyncClip.detect`
        or :meth:`~motion_sync.synced_dataset.SyncClip.attach_contact`.

        Args:
            clip: Synced clip with required bodies/markers for this detector.
            config: Optional detector configuration (YAML-backed in CLI).

        Returns:
            New :class:`~motion_sync.contact_layer.ContactLayer` with ``layer_id`` matching
            this type.
        """

    @abstractmethod
    def read(self, clip: Any, layer: ContactLayer) -> ViewT:
        """Construct a typed view over an attached layer.

        Args:
            clip: Clip the layer belongs to (for time alignment and registration).
            layer: Persisted layer with matching :attr:`layer_id`.

        Returns:
            Project-specific view (e.g. per-foot state arrays).
        """


@dataclass(frozen=True)
class ContactSchema:
    """Bundle of contact types for a project (e.g. skateboarding).

    Attributes:
        types (tuple[ContactType[Any, Any], ...]): Detectors to register together.
    """

    types: tuple[ContactType[Any, Any], ...]

    def __iter__(self):
        """Iterate registered contact types."""
        return iter(self.types)


def merge_registered_contacts(
    existing: dict[str, ContactType[Any, Any]] | None,
    *contact_types: ContactType[Any, Any],
) -> dict[str, ContactType[Any, Any]]:
    """Merge contact types into a clip's registration map.

    Args:
        existing: Current ``registered_contacts`` dict, or ``None``.
        *contact_types: Types to add or reaffirm.

    Returns:
        New dict keyed by :attr:`ContactType.layer_id`.

    Raises:
        ValueError: If two different types share the same ``layer_id``.
    """
    merged = dict(existing or {})
    for contact in contact_types:
        if contact.layer_id in merged and merged[contact.layer_id] is not contact:
            raise ValueError(f"duplicate contact type layer_id {contact.layer_id!r}")
        merged[contact.layer_id] = contact
    return merged
