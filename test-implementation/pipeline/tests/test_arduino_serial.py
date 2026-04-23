"""Tests for the single-byte Arduino serial bridge.

``pyserial`` is mocked so these tests run anywhere — no hardware, no loopback.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bci_grasp.deployment.arduino_serial import GRASP_BYTE, REST_BYTE, ArduinoLink


def _make_mock_serial():
    """Patched stand-in for serial.Serial(...) — tracks bytes written."""
    mock_instance = MagicMock()
    mock_instance.is_open = True
    mock_instance.write = MagicMock()
    mock_instance.flush = MagicMock()
    mock_instance.close = MagicMock()
    return mock_instance


def test_bytes_are_single_ascii_chars():
    """The byte constants must be a single ASCII character each."""
    assert GRASP_BYTE == b"1" and len(GRASP_BYTE) == 1
    assert REST_BYTE == b"0" and len(REST_BYTE) == 1


def test_send_before_open_raises():
    link = ArduinoLink("/dev/null")
    with pytest.raises(RuntimeError, match="not open"):
        link.send("grasp")


def test_send_grasp_writes_1_byte():
    ser = _make_mock_serial()
    with patch("bci_grasp.deployment.arduino_serial.serial.Serial", return_value=ser), \
         patch("bci_grasp.deployment.arduino_serial.time.sleep"):  # skip 2s boot wait
        with ArduinoLink("/dev/null", open_delay_s=0) as link:
            link.send("grasp")
            ser.write.assert_called_once_with(GRASP_BYTE)
            ser.flush.assert_called_once()
            assert link.last_state == "grasp"


def test_send_rest_writes_0_byte():
    ser = _make_mock_serial()
    with patch("bci_grasp.deployment.arduino_serial.serial.Serial", return_value=ser), \
         patch("bci_grasp.deployment.arduino_serial.time.sleep"):
        with ArduinoLink("/dev/null") as link:
            link.send("rest")
            ser.write.assert_called_once_with(REST_BYTE)
            assert link.last_state == "rest"


def test_send_invalid_state_raises():
    ser = _make_mock_serial()
    with patch("bci_grasp.deployment.arduino_serial.serial.Serial", return_value=ser), \
         patch("bci_grasp.deployment.arduino_serial.time.sleep"):
        with ArduinoLink("/dev/null") as link:
            with pytest.raises(ValueError, match="grasp.*rest"):
                link.send("wiggle")  # type: ignore[arg-type]


def test_context_manager_closes_port():
    ser = _make_mock_serial()
    with patch("bci_grasp.deployment.arduino_serial.serial.Serial", return_value=ser), \
         patch("bci_grasp.deployment.arduino_serial.time.sleep"):
        with ArduinoLink("/dev/null") as link:
            pass
        ser.close.assert_called_once()
        assert not link.is_open


def test_context_manager_closes_on_exception():
    """Port must still close if the body of the `with` raises."""
    ser = _make_mock_serial()
    with patch("bci_grasp.deployment.arduino_serial.serial.Serial", return_value=ser), \
         patch("bci_grasp.deployment.arduino_serial.time.sleep"):
        with pytest.raises(RuntimeError, match="boom"):
            with ArduinoLink("/dev/null") as link:
                link.send("grasp")
                raise RuntimeError("boom")
        ser.close.assert_called_once()


def test_open_delay_is_respected():
    """Verifies we sleep after opening the port (Arduino DTR-reset workaround)."""
    ser = _make_mock_serial()
    with patch("bci_grasp.deployment.arduino_serial.serial.Serial", return_value=ser), \
         patch("bci_grasp.deployment.arduino_serial.time.sleep") as sleep_mock:
        with ArduinoLink("/dev/null", open_delay_s=2.0):
            pass
        sleep_mock.assert_called_once_with(2.0)
