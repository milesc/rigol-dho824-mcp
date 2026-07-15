import copy
from pathlib import Path
import struct
import tempfile
import unittest

from rigol_dho824_mcp.server import _validate_native_wfm


def _write_wfm(path):
    data_offset = 64
    raw = bytearray(data_offset + 12)
    struct.pack_into("<Q", raw, data_offset - 40, 6)
    struct.pack_into("<6H", raw, data_offset, 100, 200, 101, 201, 102, 202)
    path.write_bytes(raw)


def _metadata():
    return {
        "memory_depth_per_channel": 3,
        "channels": [
            {"channel": 1, "enabled": True, "verification_raw": [100, 101]},
            {"channel": 2, "enabled": False},
            {"channel": 3, "enabled": True, "verification_raw": [200, 201]},
        ],
    }


class NativeWfmValidationTests(unittest.TestCase):
    def test_adds_geometry_hash_and_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            wfm_path = Path(directory) / "data.wfm"
            _write_wfm(wfm_path)
            metadata = _metadata()

            _validate_native_wfm(str(wfm_path), metadata)

        self.assertEqual(metadata["wfm_bytes"], 76)
        self.assertEqual(metadata["sample_data_offset"], 64)
        self.assertEqual(metadata["sample_interleave"], [1, 3])
        self.assertEqual(len(metadata["wfm_sha256"]), 64)
        self.assertIs(metadata["channels"][0]["verification_exact_match"], True)
        self.assertEqual(metadata["channels"][0]["verification_points"], 2)
        self.assertIs(metadata["channels"][2]["verification_exact_match"], True)
        self.assertNotIn("verification_raw", metadata["channels"][0])

    def test_rejects_sample_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            wfm_path = Path(directory) / "data.wfm"
            _write_wfm(wfm_path)
            metadata = copy.deepcopy(_metadata())
            metadata["channels"][2]["verification_raw"][1] = 999

            with self.assertRaisesRegex(
                RuntimeError, "CH3 WFM verification mismatch at 1"
            ):
                _validate_native_wfm(str(wfm_path), metadata)


if __name__ == "__main__":
    unittest.main()
