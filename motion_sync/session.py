"""Bundle mocap + contact schemas for one-shot clip registration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Generic, TypeVar

from motion_sync.contact_registration import ContactSchema, ContactType
from motion_sync.mocap_schema import MocapSchema
from motion_sync.video_schema import VideoSchema

BodyT = TypeVar("BodyT", bound=StrEnum)


@dataclass(frozen=True)
class ClipSession(Generic[BodyT]):
    """Register mocap, video, and contact types on a loaded :class:`~motion_sync.synced_dataset.SyncClip`."""

    mocap: MocapSchema[BodyT] | None = None
    contacts: ContactSchema | None = None
    video: VideoSchema[Any] | None = None

    def register(self, clip: Any) -> Any:
        """Apply this session’s schemas to ``clip`` (bodies/markers/video/contacts)."""
        out = clip
        if self.mocap is not None:
            if out.markers:
                out = out.register_mocap(self.mocap)
            else:
                out = out.register_bodies(self.mocap.bodies)
        if self.video is not None:
            out = out.register_video(self.video)
        if self.contacts is not None:
            out = out.register_contacts(self.contacts)
        return out


def apply_clip_registration(
    clip: Any,
    *,
    mocap: MocapSchema[Any] | None = None,
    contacts: ContactSchema | ContactType[Any, Any] | None = None,
    video: VideoSchema[Any] | None = None,
    session: ClipSession[Any] | None = None,
) -> Any:
    """Register schemas on ``clip`` via ``session=`` and/or ``mocap`` / ``contacts`` / ``video``."""
    if session is not None and (mocap is not None or contacts is not None or video is not None):
        raise ValueError("Pass session= alone, or mocap=/contacts=/video=, not both.")
    if session is not None:
        return session.register(clip)

    out = clip
    if mocap is not None:
        if out.markers:
            out = out.register_mocap(mocap)
        else:
            out = out.register_bodies(mocap.bodies)
    if video is not None:
        out = out.register_video(video)
    if contacts is not None:
        if isinstance(contacts, ContactSchema):
            out = out.register_contacts(contacts)
        else:
            out = out.register_contacts(contacts)
    return out
