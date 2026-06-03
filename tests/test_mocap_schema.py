import unittest
from enum import StrEnum

from motion_sync.mocap_schema import MocapSchema, validate_body_enum, validate_body_marker_enums


class _Bodies(StrEnum):
    A = "A"
    B = "B"


class _AMarkers(StrEnum):
    M1 = "m1"
    M2 = "m2"


class _BMarkers(StrEnum):
    M3 = "m3"


_SCHEMA = MocapSchema(
    bodies=_Bodies,
    body_markers={
        _Bodies.A: _AMarkers,
        _Bodies.B: _BMarkers,
    },
)


class TestMocapSchema(unittest.TestCase):
    def test_validate_enums(self):
        validate_body_enum(_Bodies, ("A", "B"))

    def test_body_marker_enums_partition(self):
        stored = validate_body_marker_enums(
            _Bodies,
            _SCHEMA.body_markers,
            ("m1", "m2", "m3"),
        )
        self.assertEqual(stored["A"], ("m1", "m2"))
        self.assertEqual(stored["B"], ("m3",))

    def test_rejects_duplicate_vicon_value(self):
        class _BadBMarkers(StrEnum):
            M2 = "m2"
            M3 = "m3"

        bad = {
            _Bodies.A: _AMarkers,
            _Bodies.B: _BadBMarkers,
        }
        with self.assertRaises(ValueError):
            validate_body_marker_enums(_Bodies, bad, ("m1", "m2", "m3"))

    def test_validate_against_clip(self):
        stored = _SCHEMA.validate_against_clip(("A", "B"), ("m1", "m2", "m3"))
        self.assertEqual(len(stored), 2)


if __name__ == "__main__":
    unittest.main()
