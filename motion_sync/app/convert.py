from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.msg import get_types_from_msg

import typer
import pathlib
import numpy as np
import pandas as pd

from motion_sync import _storage

convert_app = typer.Typer(help="Commands for converting data between formats.")

def order_preserving_unique(arr):
    values, indices = np.unique(arr, return_index=True)
    return values[np.argsort(indices)]

def merge_tf_and_marker_data(tf_data, marker_data):
    tf_stamps = tf_data["header.stamp"]
    marker_stamps = marker_data["header.stamp"]

    # NOTE: tf_stamps are a subset of marker_stamps, not the other way around.
    assert np.all(np.isin(tf_stamps, marker_stamps))
    stamps = np.unique(np.concatenate([tf_stamps, marker_stamps]))  # sorted by default

    # Create a mapping from timestamp to frame index
    stamp_to_frame = {t: i for i, t in enumerate(stamps)}

    # Create a lookup table mapping the TF stamp to the available info.
    num_frames = len(stamps)

    # ['' 'Left_Shoe' 'Right_Shoe' 'Skateboard']
    subject_names = np.unique(marker_data["subject_name"])
    subject_names = subject_names[subject_names != ""]
    num_subjects = len(subject_names)
    assert num_subjects == 3

    subject_to_index = {subject: i for i, subject in enumerate(subject_names)}

    marker_names = np.unique(marker_data["marker_name"])
    marker_names = marker_names[marker_names != ""]
    num_markers = len(marker_names)
    assert num_markers == 28

    marker_to_index = {marker: i for i, marker in enumerate(marker_names)}

    # ['vicon/Skateboard/Skateboard' 'vicon/Left_Shoe/Left_Shoe' 'vicon/Right_Shoe/Right_Shoe']
    # Get all unique /tf child frame IDs in the order in which they appear
    # child_frame_ids = order_preserving_unique(tf_data["child_frame_id"])
    # num_bodies = len(child_frame_ids)
    # assert num_bodies == 3

    lookup = {
        "frame": np.arange(num_frames),
        "stamp": stamps,
        "body_names": subject_names,
        "marker_names": marker_names,
    }

    body_pos = np.full((num_frames, num_subjects, 3), np.nan, dtype=np.float32)
    body_quat = np.full((num_frames, num_subjects, 4), np.nan, dtype=np.float32)
    body_occluded = np.full((num_frames, num_subjects), True, dtype=bool)
    marker_pos = np.full((num_frames, num_markers, 3), np.nan, dtype=np.float32)
    marker_occluded = np.full((num_frames, num_markers), None, dtype=bool)

    for i in tqdm(range(len(tf_stamps))):
        t = tf_stamps[i]
        frame = stamp_to_frame[t]
        child_frame_id = tf_data["child_frame_id"][i]
        subject = child_frame_id.split("/")[1]
        if subject not in subject_names:
            raise ValueError(f"Subject {subject} not in subject_names when processing TF data")
        subject_index = subject_to_index[subject]
        pos = tf_data["xyz"][i]
        quat = tf_data["wxyz"][i]
        body_pos[frame, subject_index] = pos
        body_quat[frame, subject_index] = quat
        body_occluded[frame, subject_index] = False
    
    for i in tqdm(range(len(marker_stamps))):
        t = marker_stamps[i]
        frame = stamp_to_frame[t]
        marker = marker_data["marker_name"][i]
        subject = marker_data["subject_name"][i]
        if subject == "" or marker == "":
            continue
        if subject not in subject_names:
            raise ValueError(f"Subject {subject} not in subject_names when processing marker data")
        if marker not in marker_names:
            raise ValueError(f"Marker {marker} not in marker_names when processing marker data")
        marker_index = marker_to_index[marker]
        marker_pos[frame, marker_index] = marker_data["xyz"][i]
        marker_occluded[frame, marker_index] = marker_data["occluded"][i]

    # Count the number of None values in marker_occluded
    if np.any(marker_occluded == None):
        raise ValueError("marker_occluded contains None values")

    lookup["body_pos"] = body_pos
    lookup["body_quat"] = body_quat
    lookup["marker_pos"] = marker_pos
    lookup["marker_occluded"] = marker_occluded

    return lookup

import numpy as np
from tqdm import tqdm


def merge_tf_and_marker_data(tf_data, marker_data):
    tf_stamps = tf_data["header.stamp"]
    marker_stamps = marker_data["header.stamp"]

    # NOTE: tf_stamps are a subset of marker_stamps, not the other way around.
    assert np.all(np.isin(tf_stamps, marker_stamps))

    # Since tf_stamps are a subset of marker_stamps, this is equivalent to:
    # np.unique(np.concatenate([tf_stamps, marker_stamps]))
    stamps = np.unique(marker_stamps)

    num_frames = len(stamps)

    subject_names = np.unique(marker_data["subject_name"])
    subject_names = subject_names[subject_names != ""]
    num_subjects = len(subject_names)
    assert num_subjects == 3

    marker_names = np.unique(marker_data["marker_name"])
    marker_names = marker_names[marker_names != ""]
    num_markers = len(marker_names)
    assert num_markers == 28

    lookup = {
        "frame": np.arange(num_frames),
        "stamp": stamps,
        "body_names": subject_names,
        "marker_names": marker_names,
    }

    body_pos = np.full((num_frames, num_subjects, 3), np.nan, dtype=np.float32)
    body_quat = np.full((num_frames, num_subjects, 4), np.nan, dtype=np.float32)
    body_occluded = np.full((num_frames, num_subjects), True, dtype=bool)

    marker_pos = np.full((num_frames, num_markers, 3), np.nan, dtype=np.float32)

    # Important: your original code uses dtype=bool with fill_value=None.
    # That actually becomes False, not None. Tiny NumPy goblin behavior.
    marker_occluded = np.full((num_frames, num_markers), False, dtype=bool)

    # -----------------------------
    # Vectorized TF data assignment
    # -----------------------------

    tf_frames = np.searchsorted(stamps, tf_stamps)

    child_frame_ids = np.asarray(tf_data["child_frame_id"])
    tf_subjects = np.array([s.split("/")[1] for s in child_frame_ids])

    tf_subject_indices = np.searchsorted(subject_names, tf_subjects)

    bad_tf_subjects = \
        (tf_subject_indices >= len(subject_names)) | \
        (subject_names[tf_subject_indices] != tf_subjects)

    if np.any(bad_tf_subjects):
        bad_subject = tf_subjects[np.flatnonzero(bad_tf_subjects)[0]]
        raise ValueError(
            f"Subject {bad_subject} not in subject_names when processing TF data"
        )

    body_pos[tf_frames, tf_subject_indices] = tf_data["xyz"].astype(np.float32, copy=False)
    body_quat[tf_frames, tf_subject_indices] = tf_data["wxyz"].astype(np.float32, copy=False)
    body_occluded[tf_frames, tf_subject_indices] = False

    # --------------------------------
    # Vectorized marker data assignment
    # --------------------------------

    marker_subjects = np.asarray(marker_data["subject_name"])
    markers = np.asarray(marker_data["marker_name"])

    valid_marker_rows = (marker_subjects != "") & (markers != "")

    valid_marker_stamps = marker_stamps[valid_marker_rows]
    valid_marker_subjects = marker_subjects[valid_marker_rows]
    valid_markers = markers[valid_marker_rows]

    marker_frames = np.searchsorted(stamps, valid_marker_stamps)
    marker_indices = np.searchsorted(marker_names, valid_markers)

    bad_markers = \
        (marker_indices >= len(marker_names)) | \
        (marker_names[marker_indices] != valid_markers)

    if np.any(bad_markers):
        bad_marker = valid_markers[np.flatnonzero(bad_markers)[0]]
        raise ValueError(
            f"Marker {bad_marker} not in marker_names when processing marker data"
        )

    # This preserves the original subject validation, even though subject_index
    # is not otherwise used for marker placement.
    marker_subject_indices = np.searchsorted(subject_names, valid_marker_subjects)

    bad_marker_subjects = \
        (marker_subject_indices >= len(subject_names)) | \
        (subject_names[marker_subject_indices] != valid_marker_subjects)

    if np.any(bad_marker_subjects):
        bad_subject = valid_marker_subjects[np.flatnonzero(bad_marker_subjects)[0]]
        raise ValueError(
            f"Subject {bad_subject} not in subject_names when processing marker data"
        )

    marker_pos[marker_frames, marker_indices] = marker_data["xyz"][valid_marker_rows].astype(
        np.float32,
        copy=False,
    )

    marker_occluded[marker_frames, marker_indices] = marker_data["occluded"][valid_marker_rows]

    # This check is redundant because marker_occluded is bool dtype.
    # Kept to match the original control flow vibe, because apparently we enjoy rituals.
    if np.any(marker_occluded == None):
        raise ValueError("marker_occluded contains None values")

    lookup["body_pos"] = body_pos
    lookup["body_quat"] = body_quat
    lookup["marker_pos"] = marker_pos
    lookup["marker_occluded"] = marker_occluded
    lookup["body_occluded"] = body_occluded

    return lookup

@convert_app.command(help="Convert a ROS2 bag to CSV and NPZ files.")
def bag(bag_path: Path, output_dir: Path):
    # Ensure the output path is a directory
    if not output_dir.is_dir():
        raise ValueError(f"Output path '{output_dir}' is not a directory")

    # Create the output directory for the bag
    output_dir = output_dir / bag_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a typestore and register the message types from the Vicon ROS bridge
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    for msg in pathlib.Path("data/msg").glob("*.msg"):
        with open(msg, "r") as file:
            msg_str = file.read()
            name = "vicon_bridge/msg/" + msg.stem
            typestore.register(get_types_from_msg(msg_str, name))

    with Reader(bag_path) as reader:
        rows = []
        rows_per_topic = defaultdict(list)

        # Iterate over messages.
        for connection, timestamp, rawdata in reader.messages():
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            row = {
                "topic": connection.topic,
                "msgtype": connection.msgtype,
                "timestamp": timestamp / 1e9,
                "data": msg,
            }

            if hasattr(msg, "header"):
                header = msg.header
                # row["header.seq"] = header.seq if hasattr(header, "seq") else None
                # row["header.frame_id"] = header.frame_id if hasattr(header, "frame_id") else None

                if hasattr(header, "stamp"):
                    stamp = header.stamp
                    row["header.stamp"] = stamp.sec + stamp.nanosec / 1e9
                else:
                    row["header.stamp"] = None
            else:
                row["header.stamp"] = None
                # row["header.seq"] = None
                # row["header.frame_id"] = None

            rows.append(row)
            rows_per_topic[connection.topic].append(row)

        for topic, rows_by_topic in rows_per_topic.items():
            # Replace the data column with columns for each field in the message
            new_rows_by_topic = []
            for row in rows_by_topic:
                row_copy = row.copy()
                row_copy.pop("data")

                if hasattr(row["data"], "frame_number"):
                    row_copy["frame_number"] = row["data"].frame_number

                for field, value in row["data"].__dict__.items():
                    if field in {"header", "__msgtype__", "frame_number"}:
                        continue
                    elif field == "translation":
                        translation_row = row_copy.copy()
                        translation_row[field] = value.x, value.y, value.z
                        new_rows_by_topic.append(translation_row)
                    elif field == "markers":
                        for marker in value:
                            marker_row = row_copy.copy()
                            marker_row["marker_name"] = marker.marker_name
                            marker_row["subject_name"] = marker.subject_name
                            marker_row["segment_name"] = marker.segment_name
                            marker_row["occluded"] = marker.occluded
                            if marker.occluded:
                                marker_row["x"] = None
                                marker_row["y"] = None
                                marker_row["z"] = None
                            else:
                                marker_row["x"] = marker.translation.x / 1000
                                marker_row["y"] = marker.translation.y / 1000
                                marker_row["z"] = marker.translation.z / 1000
                            new_rows_by_topic.append(marker_row)
                    elif field == "transforms":
                        for transform in value:
                            transform_row = row_copy.copy()
                            transform_row["frame_id"] = transform.header.frame_id
                            transform_row["child_frame_id"] = transform.child_frame_id
                            transform_row["x"] = transform.transform.translation.x
                            transform_row["y"] = transform.transform.translation.y
                            transform_row["z"] = transform.transform.translation.z
                            transform_row["qx"] = transform.transform.rotation.x
                            transform_row["qy"] = transform.transform.rotation.y
                            transform_row["qz"] = transform.transform.rotation.z
                            transform_row["qw"] = transform.transform.rotation.w
                            assert transform_row["header.stamp"] is None
                            if hasattr(transform.header, "stamp"):
                                stamp = transform.header.stamp
                                transform_row["header.stamp"] = stamp.sec + stamp.nanosec / 1e9
                            else:
                                transform_row["header.stamp"] = None
                            # transform_row["header.seq"] = transform.header.seq if hasattr(transform.header, "seq") else None
                            new_rows_by_topic.append(transform_row)
                    else:
                        raise ValueError(f"Unknown field: {field}")

            # Save the filtered topic messages to a separate CSV file
            csv_path = output_dir / f"{topic}.csv".strip('/')
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(new_rows_by_topic)
            df.to_csv(csv_path, index=False)

            # Convert the CSV file to a NPZ file
            # Get each column as a numpy array using a list comprehension
            npz_data = {col: df[col].values for col in df.columns}
            if "x" in npz_data and "y" in npz_data and "z" in npz_data:
                npz_data["xyz"] = np.column_stack((npz_data["x"], npz_data["y"], npz_data["z"]))
                npz_data.pop("x")
                npz_data.pop("y")
                npz_data.pop("z")
            if "qx" in npz_data and "qy" in npz_data and "qz" in npz_data and "qw" in npz_data:
                npz_data["wxyz"] = np.column_stack((npz_data["qw"], npz_data["qx"], npz_data["qy"], npz_data["qz"]))
                npz_data.pop("qx")
                npz_data.pop("qy")
                npz_data.pop("qz")
                npz_data.pop("qw")
            npz_data["length"] = len(df)
            np.savez(csv_path.with_suffix(".npz"), **npz_data)

        # Save messages.csv
        csv_path = output_dir / "messages.csv"

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        
        # Convert the CSV file to a NPZ file
        # Get each column as a numpy array using a list comprehension
        # npz_data = {col: df[col].values for col in df.columns}
        # np.savez(csv_path.with_suffix(".npz"), **npz_data)
    
    # Merge the TF and marker data
    try:
        tf_data = np.load(output_dir / "tf.npz", allow_pickle=True)
        marker_data = np.load(output_dir / "vicon" / "markers.npz", allow_pickle=True)
        payload = merge_tf_and_marker_data(tf_data, marker_data)
        _storage.write_vicon_mocap(_storage.vicon_mocap_path(output_dir), payload)
    except Exception as e:
        print(f"Error merging TF and marker data: {e}")