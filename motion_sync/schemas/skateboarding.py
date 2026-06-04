"""Two-foot skate Vicon vocabulary (matches ``configs/motion_sync.yaml``)."""

from __future__ import annotations

from enum import StrEnum

from motion_sync.contact_model import patch_calibration, rigid_body_contact_model
from motion_sync.contact_registration import ContactSchema
from motion_sync.contacts.foot_support import FootSupport
from motion_sync.contacts.shoe_board_grip import ShoeBoardGrip
from motion_sync.mocap_schema import MocapSchema
from motion_sync.session import ClipSession
from motion_sync.video_schema import VideoSchema


class Bodies(StrEnum):
    """Vicon rigid-body names for skate trials.

    Values are the strings stored in ``vicon.npz`` / ``synced.npz`` body tables.

    Attributes:
        LEFT_SHOE (str): Left shoe rigid body (``"Left_Shoe"``).
        RIGHT_SHOE (str): Right shoe rigid body (``"Right_Shoe"``).
        SKATEBOARD (str): Skateboard rigid body (``"Skateboard"``).
    """

    LEFT_SHOE = "Left_Shoe"
    RIGHT_SHOE = "Right_Shoe"
    SKATEBOARD = "Skateboard"


class LeftShoeMarkers(StrEnum):
    """Left shoe Vicon marker labels.

    Logical names (``HEEL``, ``TOE_CENTER``, …) are shared with
    :class:`RightShoeMarkers`; values are unique Vicon strings.

    Attributes:
        TOE_CENTER (str): Center of forefoot cluster.
        LITTLE_TOE (str): Lateral toe marker.
        LATERAL_ARCH (str): Lateral midfoot marker.
        MEDIAL_ARCH (str): Medial midfoot marker.
        BIG_TOE (str): Medial toe marker.
        HEEL (str): Heel marker.
    """

    TOE_CENTER = "Unlabeled18658"
    LITTLE_TOE = "Unlabeled19573"
    LATERAL_ARCH = "Unlabeled20169"
    MEDIAL_ARCH = "Unlabeled20174"
    BIG_TOE = "Unlabeled20175"
    HEEL = "Unlabeled20171"


class RightShoeMarkers(StrEnum):
    """Right shoe Vicon marker labels.

    Same logical names as :class:`LeftShoeMarkers` with different Vicon values.

    Attributes:
        LITTLE_TOE (str): Lateral toe marker.
        LATERAL_ARCH (str): Lateral midfoot marker.
        TOE_CENTER (str): Center of forefoot cluster.
        BIG_TOE (str): Medial toe marker.
        MEDIAL_ARCH (str): Medial midfoot marker.
        HEEL (str): Heel marker.
    """

    LITTLE_TOE = "Unlabeled18579"
    LATERAL_ARCH = "Unlabeled20165"
    TOE_CENTER = "Unlabeled20991"
    BIG_TOE = "Unlabeled21322"
    MEDIAL_ARCH = "Unlabeled21324"
    HEEL = "Unlabeled21323"


SOLE_PATCH = "sole"
"""Canonical patch name for shoe sole contact surfaces."""

LEFT_SHOE_CONTACT_MODEL = rigid_body_contact_model(
    Bodies.LEFT_SHOE,
    marker_type=LeftShoeMarkers,
    patches={
        SOLE_PATCH: patch_calibration(
            (
                LeftShoeMarkers.HEEL,
                LeftShoeMarkers.TOE_CENTER,
                LeftShoeMarkers.BIG_TOE,
                LeftShoeMarkers.LITTLE_TOE,
            )
        ),
    },
)
"""Left shoe body-local contact model."""

RIGHT_SHOE_CONTACT_MODEL = rigid_body_contact_model(
    Bodies.RIGHT_SHOE,
    marker_type=RightShoeMarkers,
    patches={
        SOLE_PATCH: patch_calibration(
            (
                RightShoeMarkers.HEEL,
                RightShoeMarkers.TOE_CENTER,
                RightShoeMarkers.BIG_TOE,
                RightShoeMarkers.LITTLE_TOE,
            )
        ),
    },
)
"""Right shoe body-local contact model."""

SKATE_CONTACT_MODEL = (LEFT_SHOE_CONTACT_MODEL, RIGHT_SHOE_CONTACT_MODEL)
"""Body-local contact models used by skate foot-support detection."""


class SmplxCoreJoints(StrEnum):
    """SMPL-X core joint names for GVHMR / video-side kinematics.

    Enum declaration order matches :meth:`~motion_sync.synced_dataset.SyncClip.core_joint_positions`
    column order.

    Attributes:
        PELVIS (str): Root pelvis joint.
        L_HIP (str): Left hip.
        R_HIP (str): Right hip.
        SPINE1 (str): Lower spine.
        L_KNEE (str): Left knee.
        R_KNEE (str): Right knee.
        SPINE2 (str): Mid spine.
        L_ANKLE (str): Left ankle.
        R_ANKLE (str): Right ankle.
        SPINE3 (str): Upper spine.
        L_FOOT (str): Left foot.
        R_FOOT (str): Right foot.
        NECK (str): Neck.
        HEAD (str): Head.
        L_SHOULDER (str): Left shoulder.
        R_SHOULDER (str): Right shoulder.
        L_ELBOW (str): Left elbow.
        R_ELBOW (str): Right elbow.
        L_WRIST (str): Left wrist.
        R_WRIST (str): Right wrist.
    """

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
    """Skateboard Vicon marker labels (poles, wheels, deck).

    Attributes:
        FRONT_RIGHT_POLE_LEFT (str): Front-right truck pole, left-side marker.
        REAR_RIGHT_POLE_RIGHT (str): Rear-right truck pole, right-side marker.
        FRONT_LEFT_POLE_RIGHT (str): Front-left truck pole, right-side marker.
        FRONT_RIGHT_POLE_RIGHT (str): Front-right truck pole, right-side marker.
        FRONT_LEFT_POLE_LEFT (str): Front-left truck pole, left-side marker.
        REAR_RIGHT_POLE_LEFT (str): Rear-right truck pole, left-side marker.
        REAR_LEFT_POLE_LEFT (str): Rear-left truck pole, left-side marker.
        REAR_LEFT_POLE_RIGHT (str): Rear-left truck pole, right-side marker.
        FRONT_LEFT_WHEEL (str): Front-left wheel marker.
        FRONT_RIGHT_WHEEL (str): Front-right wheel marker.
        REAR_LEFT_WHEEL (str): Rear-left wheel marker.
        REAR_RIGHT_WHEEL (str): Rear-right wheel marker.
        REAR_LEFT_BOARD (str): Rear-left deck marker.
        REAR_RIGHT_BOARD (str): Rear-right deck marker.
        FRONT_LEFT_BOARD (str): Front-left deck marker.
        FRONT_CENTER_BOARD (str): Front-center deck marker.
    """

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
"""Registered mocap schema: shoes + skateboard bodies and marker enums."""

SKATE_FOOT_SUPPORT = FootSupport(
    Bodies.LEFT_SHOE,
    Bodies.RIGHT_SHOE,
    Bodies.SKATEBOARD,
    contact_models=SKATE_CONTACT_MODEL,
    sole_patch_names={
        Bodies.LEFT_SHOE.value: SOLE_PATCH,
        Bodies.RIGHT_SHOE.value: SOLE_PATCH,
    },
)
"""Foot-support contact detector for left/right shoes and board."""

SKATE_SHOE_BOARD_GRIP = ShoeBoardGrip(
    left=Bodies.LEFT_SHOE,
    right=Bodies.RIGHT_SHOE,
    foot_support=SKATE_FOOT_SUPPORT,
)
"""Binary shoe-on-board grip derived from :data:`SKATE_FOOT_SUPPORT`."""

SKATE_CONTACTS = ContactSchema(types=(SKATE_FOOT_SUPPORT, SKATE_SHOE_BOARD_GRIP))
"""Contact types registered for skate sessions."""

SKATE_VIDEO = VideoSchema.smplx_core(SmplxCoreJoints)
"""Video-side SMPL-X core joint schema for sync and visualization."""

SKATE_SESSION = ClipSession(mocap=SKATE_MOCAP, contacts=SKATE_CONTACTS, video=SKATE_VIDEO)
"""Default :class:`~motion_sync.session.ClipSession` for skateboarding trials."""
