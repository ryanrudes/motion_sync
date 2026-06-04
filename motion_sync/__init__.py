"""Motion sync: Vicon + GVHMR ingest, time sync, and typed SyncClip access."""

from motion_sync.body_marker_plot import draw_body_markers_frame, plot_body_markers
from motion_sync.contact_layer import ContactLayer
from motion_sync.contact_model import body_name, marker_name, patch_calibration, rigid_body_contact_model
from motion_sync.contact_registration import ContactSchema, ContactType
from motion_sync.mocap_schema import MocapSchema, validate_body_enum
from motion_sync.session import ClipSession
from motion_sync.synced_dataset import (
    AxisConvention,
    JointTrack,
    MarkerTracks,
    QuaternionOrder,
    RigidBodyPose,
    RigidBodyTrack,
    SyncClip,
    SyncMetadata,
    ViconMocap,
    VideoSmplx,
)
from motion_sync.vicon_recording import ViconRecording
from motion_sync.video_schema import VideoSchema

__all__ = [
    "AxisConvention",
    "ClipSession",
    "ContactLayer",
    "ContactSchema",
    "ContactType",
    "JointTrack",
    "MarkerTracks",
    "MocapSchema",
    "QuaternionOrder",
    "RigidBodyPose",
    "RigidBodyTrack",
    "SyncClip",
    "SyncMetadata",
    "ViconMocap",
    "ViconRecording",
    "VideoSchema",
    "VideoSmplx",
    "body_name",
    "draw_body_markers_frame",
    "marker_name",
    "patch_calibration",
    "plot_body_markers",
    "rigid_body_contact_model",
    "validate_body_enum",
]
