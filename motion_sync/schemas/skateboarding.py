"""Two-foot skate Vicon vocabulary (matches ``configs/motion_sync.yaml``)."""

from __future__ import annotations

from enum import StrEnum

from motion_sync.contact_registration import ContactSchema
from motion_sync.contacts.foot_support import FootSupport
from motion_sync.contacts.shoe_board_grip import ShoeBoardGrip
from motion_sync.mocap_schema import MocapSchema
from motion_sync.session import ClipSession
from motion_sync.video_schema import VideoSchema


class Bodies(StrEnum):
    LEFT_SHOE = "Left_Shoe"
    RIGHT_SHOE = "Right_Shoe"
    SKATEBOARD = "Skateboard"


class LeftShoeMarkers(StrEnum):
    """Left shoe markers (Vicon strings as values)."""

    TOE_CENTER = "Unlabeled18658"
    LITTLE_TOE = "Unlabeled19573"
    LATERAL_ARCH = "Unlabeled20169"
    MEDIAL_ARCH = "Unlabeled20174"
    BIG_TOE = "Unlabeled20175"
    HEEL = "Unlabeled20171"


class RightShoeMarkers(StrEnum):
    """Right shoe markers — same logical names as left, different Vicon values."""

    LITTLE_TOE = "Unlabeled18579"
    LATERAL_ARCH = "Unlabeled20165"
    TOE_CENTER = "Unlabeled20991"
    BIG_TOE = "Unlabeled21322"
    MEDIAL_ARCH = "Unlabeled21324"
    HEEL = "Unlabeled21323"


class SmplxCoreJoints(StrEnum):
    """Retarget ``smplx`` core joints (enum order = :meth:`SyncClip.core_joint_positions` columns)."""

    PELVIS = "Pelvis"
    L_HIP = "L_Hip"
    R_HIP = "R_Hip"
    SPINE1 = "Spine1"
    L_KNEE = "L_Knee"
    R_KNEE = "R_Knee"
    SPINE2 = "Spine2"
    L_ANKLE = "L_Ankle"
    R_ANKLE = "R_Ankle"
    SPINE3 = "Spine3"
    L_FOOT = "L_Foot"
    R_FOOT = "R_Foot"
    NECK = "Neck"
    HEAD = "Head"
    L_SHOULDER = "L_Shoulder"
    R_SHOULDER = "R_Shoulder"
    L_ELBOW = "L_Elbow"
    R_ELBOW = "R_Elbow"
    L_WRIST = "L_Wrist"
    R_WRIST = "R_Wrist"


class SkateboardMarkers(StrEnum):
    FRONT_RIGHT_POLE_LEFT = "front_right_pole_left"
    REAR_RIGHT_POLE_RIGHT = "rear_right_pole_right"
    FRONT_LEFT_POLE_RIGHT = "front_left_pole_right"
    FRONT_RIGHT_POLE_RIGHT = "front_right_pole_right"
    FRONT_LEFT_POLE_LEFT = "front_left_pole_left"
    REAR_RIGHT_POLE_LEFT = "rear_right_pole_left"
    REAR_LEFT_POLE_LEFT = "rear_left_pole_left"
    REAR_LEFT_POLE_RIGHT = "rear_left_pole_right"
    FRONT_LEFT_WHEEL = "front_left_wheel"
    FRONT_RIGHT_WHEEL = "front_right_wheel"
    REAR_LEFT_WHEEL = "rear_left_wheel"
    REAR_RIGHT_WHEEL = "rear_right_wheel"
    REAR_LEFT_BOARD = "rear_left_board"
    REAR_RIGHT_BOARD = "rear_right_board"
    FRONT_LEFT_BOARD = "front_left_board"
    FRONT_CENTER_BOARD = "front_center_board"


SKATE_MOCAP: MocapSchema[Bodies] = MocapSchema(
    bodies=Bodies,
    body_markers={
        Bodies.LEFT_SHOE: LeftShoeMarkers,
        Bodies.RIGHT_SHOE: RightShoeMarkers,
        Bodies.SKATEBOARD: SkateboardMarkers,
    },
)

SKATE_FOOT_SUPPORT = FootSupport(
    left=Bodies.LEFT_SHOE,
    right=Bodies.RIGHT_SHOE,
    board=Bodies.SKATEBOARD,
)

SKATE_SHOE_BOARD_GRIP = ShoeBoardGrip(
    left=Bodies.LEFT_SHOE,
    right=Bodies.RIGHT_SHOE,
    foot_support=SKATE_FOOT_SUPPORT,
)

SKATE_CONTACTS = ContactSchema(types=(SKATE_FOOT_SUPPORT, SKATE_SHOE_BOARD_GRIP))

SKATE_VIDEO = VideoSchema.smplx_core(SmplxCoreJoints)

SKATE_SESSION = ClipSession(mocap=SKATE_MOCAP, contacts=SKATE_CONTACTS, video=SKATE_VIDEO)
