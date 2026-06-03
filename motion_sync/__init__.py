"""Motion sync: Vicon + GVHMR ingest, time sync, and typed SyncClip access."""

from motion_sync.body_marker_plot import draw_body_markers_frame, plot_body_markers
from motion_sync.contact_layer import ContactLayer
from motion_sync.contact_registration import ContactSchema, ContactType
from motion_sync.session import ClipSession
from motion_sync.mocap_schema import MocapSchema, validate_body_enum
from motion_sync.synced_dataset import (
    AxisConvention,
    JointTrack,
    MarkerTracks,
    QuaternionOrder,
    RigidBodyPose,
    RigidBodyTrack,
    SyncClip,
    SyncMetadata,
    VideoSmplx,
    ViconMocap,
)
from motion_sync.video_schema import VideoSchema
from motion_sync.vicon_recording import ViconRecording

__all__ = [
    "ContactLayer",
    "ClipSession",
    "ContactSchema",
    "ContactType",
    "draw_body_markers_frame",
    "plot_body_markers",
    "MocapSchema",
    "AxisConvention",
    "JointTrack",
    "MarkerTracks",
    "VideoSchema",
    "QuaternionOrder",
    "RigidBodyPose",
    "RigidBodyTrack",
    "SyncClip",
    "SyncMetadata",
    "VideoSmplx",
    "ViconMocap",
    "ViconRecording",
    "validate_body_enum",
]
