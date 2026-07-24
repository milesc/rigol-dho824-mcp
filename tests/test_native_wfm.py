import copy
from pathlib import Path
import struct
import tempfile
import unittest

from pydantic import ValidationError

from rigol_dho824_mcp.server import (
    CaptureEdgeTriggerSetup,
    CapturePulseTriggerSetup,
    CaptureTimebaseSetup,
    TriggerSlope,
    TriggerSweep,
    _normalize_pulse_polarity,
    _normalize_pulse_width_condition,
    _pulse_width_write_commands,
    _validate_native_wfm,
)


def _write_wfm(path):
    data_offset = 64
    raw = bytearray(data_offset + 12)
    struct.pack_into("<Q", raw, data_offset - 40, 6)
    struct.pack_into("<6H", raw, data_offset, 100, 200, 101, 201, 102, 202)
    path.write_bytes(raw)


def _write_three_channel_padded_wfm(path):
    data_offset = 64
    raw = bytearray(data_offset + 24)
    struct.pack_into("<Q", raw, data_offset - 40, 12)
    struct.pack_into(
        "<12H",
        raw,
        data_offset,
        100, 200, 300, 999,
        101, 201, 301, 998,
        102, 202, 302, 997,
    )
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
    def test_complete_timebase_defaults_auto_roll_off(self):
        setup = CaptureTimebaseSetup(time_per_div=0.2, time_offset=1.8)

        self.assertIs(setup.auto_roll_enabled, False)
        self.assertIs(setup.model_dump()["auto_roll_enabled"], False)

    def test_normalizes_pulse_trigger_readbacks(self):
        self.assertEqual(_normalize_pulse_polarity("NEG"), "NEGATIVE")
        self.assertEqual(_normalize_pulse_polarity("POSitive"), "POSITIVE")
        self.assertEqual(_normalize_pulse_width_condition("GRE"), "GREATER")
        self.assertEqual(_normalize_pulse_width_condition("LESS"), "LESS")
        self.assertEqual(_normalize_pulse_width_condition("GLES"), "WITHIN")

    def test_rejects_unknown_pulse_trigger_readbacks(self):
        with self.assertRaises(ValueError):
            _normalize_pulse_polarity("MAYBE")
        with self.assertRaises(ValueError):
            _normalize_pulse_width_condition("MAYBE")

    def test_greater_pulse_uses_lower_width_register(self):
        self.assertEqual(
            _pulse_width_write_commands("GREATER", 2.5e-6, None),
            [(":TRIG:PULS:LWID", 2.5e-6)],
        )

    def test_pulse_capture_setup_validates_active_width_limits(self):
        setup = CapturePulseTriggerSetup(
            channel=4,
            trigger_level=-0.1,
            pulse_polarity="NEGATIVE",
            pulse_width_condition="GREATER",
            pulse_lower_width=2.5e-6,
        )
        self.assertEqual(setup.pulse_lower_width, 2.5e-6)

        with self.assertRaises(ValidationError):
            CapturePulseTriggerSetup(
                channel=4,
                trigger_level=-0.1,
                pulse_polarity="NEGATIVE",
                pulse_width_condition="GREATER",
                pulse_upper_width=2.5e-6,
            )

        with self.assertRaises(ValidationError):
            CapturePulseTriggerSetup(
                channel=4,
                trigger_level=-0.1,
                pulse_polarity="NEGATIVE",
                pulse_width_condition="WITHIN",
                pulse_lower_width=3e-6,
                pulse_upper_width=2e-6,
            )

    def test_adds_geometry_hash_and_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            wfm_path = Path(directory) / "data.wfm"
            _write_wfm(wfm_path)
            metadata = _metadata()

            _validate_native_wfm(str(wfm_path), metadata)

        self.assertEqual(metadata["wfm_bytes"], 76)
        self.assertEqual(metadata["sample_data_offset"], 64)
        self.assertEqual(metadata["sample_interleave"], [1, 3])
        self.assertEqual(metadata["sample_storage_interleave"], [1, 3])
        self.assertEqual(len(metadata["wfm_sha256"]), 64)
        self.assertIs(metadata["channels"][0]["verification_exact_match"], True)
        self.assertEqual(metadata["channels"][0]["verification_points"], 2)
        self.assertIs(metadata["channels"][2]["verification_exact_match"], True)
        self.assertNotIn("verification_raw", metadata["channels"][0])

    def test_accepts_three_enabled_channels_in_four_storage_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            wfm_path = Path(directory) / "data.wfm"
            _write_three_channel_padded_wfm(wfm_path)
            metadata = {
                "memory_depth_per_channel": 3,
                "channels": [
                    {"channel": 1, "enabled": True, "verification_raw": [100, 101]},
                    {"channel": 2, "enabled": True, "verification_raw": [200, 201]},
                    {"channel": 3, "enabled": True, "verification_raw": [300, 301]},
                    {"channel": 4, "enabled": False},
                ],
            }

            _validate_native_wfm(str(wfm_path), metadata)

        self.assertEqual(metadata["sample_data_offset"], 64)
        self.assertEqual(metadata["sample_interleave"], [1, 2, 3])
        self.assertEqual(metadata["sample_storage_interleave"], [1, 2, 3, None])
        self.assertTrue(
            all(
                entry.get("verification_exact_match", False)
                for entry in metadata["channels"]
                if entry["enabled"]
            )
        )

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

    def test_edge_trigger_setup_accepts_documented_minimum_holdoff(self):
        setup = CaptureEdgeTriggerSetup(
            channel=1,
            trigger_level=1.0,
            trigger_slope=TriggerSlope.POSITIVE,
            trigger_sweep="NORMAL",
            holdoff_time=8e-9,
        )

        self.assertEqual(setup.trigger_sweep, TriggerSweep.NORMAL)
        self.assertEqual(setup.holdoff_time, 8e-9)

    def test_edge_trigger_setup_rejects_subminimum_holdoff(self):
        with self.assertRaises(ValidationError):
            CaptureEdgeTriggerSetup(
                channel=1,
                trigger_level=1.0,
                trigger_slope=TriggerSlope.POSITIVE,
                holdoff_time=7e-9,
            )


if __name__ == "__main__":
    unittest.main()
