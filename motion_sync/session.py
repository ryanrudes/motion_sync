"""Bundle mocap + contact schemas for one-shot clip registration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from motion_sync.contact_registration import ContactSchema, ContactType
from motion_sync.mocap_schema import MocapSchema
from motion_sync.video_schema import VideoSchema

if TYPE_CHECKING:
    from motion_sync.synced_dataset import SyncClip

BodyT = TypeVar("BodyT", bound=StrEnum)


@dataclass(frozen=True)
class ClipSession(Generic[BodyT]):
    """Register mocap, video, and contact types on a loaded :class:`~motion_sync.synced_dataset.SyncClip`.

    Typical use: define a module-level session (e.g. ``SKATE_SESSION``) and pass it to
    :meth:`SyncClip.load` so bodies, markers, joints, and contacts are wired in one step.

    Attributes:
        mocap (MocapSchema[BodyT] | None): Body and per-body marker enums; optional.
        contacts (ContactSchema | None): Bundle of :class:`~motion_sync.contact_registration.ContactType` instances.
        video (VideoSchema[Any] | None): SMPL-X / video joint schema for :meth:`SyncClip.joint`.
    """

    mocap: MocapSchema[BodyT] | None = None
    contacts: ContactSchema | None = None
    video: VideoSchema[Any] | None = None

    def register(self, clip: SyncClip[Any]) -> SyncClip[Any]:
        """Apply this session's schemas to ``clip``.

        Registers bodies (and markers when present), video joints, and contact types.
        Marker registration is skipped when the clip has no marker channel.

        Args:
            clip: Loaded :class:`~motion_sync.synced_dataset.SyncClip`.

        Returns:
            The same clip with registration fields populated.
        """
        out = clip
        if self.mocap is not None:
            out = (
                out.register_mocap(self.mocap)
                if out.markers
                else out.register_bodies(self.mocap.bodies)
            )
        if self.video is not None:
            out = out.register_video(self.video)
        if self.contacts is not None:
            out = out.register_contacts(self.contacts)
        return out


def apply_clip_registration(
    clip: SyncClip[Any],
    *,
    mocap: MocapSchema[Any] | None = None,
    contacts: ContactSchema | ContactType[Any, Any] | None = None,
    video: VideoSchema[Any] | None = None,
    session: ClipSession[Any] | None = None,
) -> SyncClip[Any]:
    """Register schemas on ``clip`` via ``session=`` and/or individual schema arguments.

    Args:
        clip: Loaded synced clip.
        mocap: Optional :class:`~motion_sync.mocap_schema.MocapSchema`.
        contacts: Optional :class:`~motion_sync.contact_registration.ContactSchema` or single
            :class:`~motion_sync.contact_registration.ContactType`.
        video: Optional :class:`~motion_sync.video_schema.VideoSchema`.
        session: Optional bundled registration; must not be combined with ``mocap`` /
            ``contacts`` / ``video``.

    Returns:
        Clip with registration applied.

    Raises:
        ValueError: If ``session`` is passed together with ``mocap``, ``contacts``, or ``video``.
    """
    if session is not None and (mocap is not None or contacts is not None or video is not None):
        raise ValueError("Pass session= alone, or mocap=/contacts=/video=, not both.")
    if session is not None:
        return session.register(clip)

    out = clip
    if mocap is not None:
        out = (
            out.register_mocap(mocap)
            if out.markers
            else out.register_bodies(mocap.bodies)
        )
    if video is not None:
        out = out.register_video(video)
    if contacts is not None:
        if isinstance(contacts, ContactSchema):
            out = out.register_contacts(contacts)
        else:
            out = out.register_contacts(contacts)
    return out
