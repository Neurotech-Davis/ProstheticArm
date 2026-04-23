"""Tests for the Python<->Arduino binary serial protocol.

Covers:
  - CRC-8 reference vectors (canonical test vector "123456789" → 0xF4).
  - encode() frame structure for every Cmd value.
  - decode() round-trips.
  - decode() raises on bad sync / bad CRC / short frame / unknown CMD.
  - find_frame() recovers from leading noise.
"""

from __future__ import annotations

import pytest

from bci_grasp.deployment import protocol
from bci_grasp.deployment.protocol import Cmd, FrameError, crc8, decode, encode, find_frame


# ---------- CRC-8 -----------------------------------------------------------


def test_crc8_empty():
    """CRC of an empty message is the init value (0x00)."""
    assert crc8(b"") == 0x00


def test_crc8_reference_vector():
    """Canonical CRC-8 (poly 0x07) vector: 'CRC-8/SMBUS' of '123456789' == 0xF4."""
    assert crc8(b"123456789") == 0xF4


def test_crc8_determinism():
    """CRC must depend only on the input bytes."""
    msg = bytes([Cmd.GRASP, 42])
    assert crc8(msg) == crc8(msg)


# ---------- encode() --------------------------------------------------------


@pytest.mark.parametrize("cmd", list(Cmd))
def test_encode_frame_shape(cmd: Cmd):
    """Every Cmd produces a 5-byte frame with correct sync + CRC."""
    frame = encode(cmd, arg=0)
    assert len(frame) == protocol.FRAME_LEN == 5
    assert frame[0] == protocol.SYNC0 == 0xAA
    assert frame[1] == protocol.SYNC1 == 0x55
    assert frame[2] == int(cmd)
    assert frame[3] == 0
    assert frame[4] == crc8(bytes([int(cmd), 0]))


def test_encode_arg_carried_through():
    """ARG byte must appear in the frame exactly and contribute to CRC."""
    frame = encode(Cmd.GRASP, arg=128)
    assert frame[3] == 128
    assert frame[4] == crc8(bytes([int(Cmd.GRASP), 128]))


@pytest.mark.parametrize("bad_arg", [-1, 256, 1000])
def test_encode_rejects_out_of_range_arg(bad_arg: int):
    with pytest.raises(ValueError):
        encode(Cmd.GRASP, arg=bad_arg)


def test_encode_rejects_out_of_range_cmd():
    with pytest.raises(ValueError):
        encode(300, arg=0)


# ---------- decode() --------------------------------------------------------


@pytest.mark.parametrize("cmd", list(Cmd))
@pytest.mark.parametrize("arg", [0, 1, 42, 255])
def test_encode_decode_roundtrip(cmd: Cmd, arg: int):
    got_cmd, got_arg = decode(encode(cmd, arg))
    assert got_cmd == cmd
    assert got_arg == arg


def test_decode_rejects_wrong_length():
    with pytest.raises(FrameError, match="expected 5 bytes"):
        decode(bytes([0xAA, 0x55, 0x01]))


def test_decode_rejects_bad_sync():
    frame = bytearray(encode(Cmd.GRASP, 0))
    frame[0] = 0x00
    with pytest.raises(FrameError, match="bad sync"):
        decode(bytes(frame))


def test_decode_rejects_bad_crc():
    frame = bytearray(encode(Cmd.GRASP, 0))
    frame[4] ^= 0xFF  # corrupt CRC
    with pytest.raises(FrameError, match="bad CRC"):
        decode(bytes(frame))


def test_decode_rejects_unknown_cmd():
    cmd_byte = 0x77  # not in Cmd enum
    arg = 0
    bad = bytes([protocol.SYNC0, protocol.SYNC1, cmd_byte, arg, crc8(bytes([cmd_byte, arg]))])
    with pytest.raises(FrameError, match="unknown CMD"):
        decode(bad)


# ---------- find_frame() ----------------------------------------------------


def test_find_frame_recovers_after_noise():
    """Leading junk bytes should not prevent a valid frame from being found."""
    good = encode(Cmd.REST, arg=7)
    stream = bytes([0x00, 0xFF, 0xAB]) + good
    result = find_frame(stream)
    assert result is not None
    cmd, arg, end = result
    assert cmd == Cmd.REST
    assert arg == 7
    assert end == len(stream)  # consumed through end of the good frame


def test_find_frame_returns_none_when_no_valid_frame():
    assert find_frame(b"\x00\x01\x02\x03\x04\x05") is None


def test_find_frame_skips_false_sync():
    """A sync-looking pair whose CRC is wrong must not be accepted; the real frame after it should."""
    fake = bytes([protocol.SYNC0, protocol.SYNC1, 0xFF, 0xFF, 0x00])  # bad CRC
    good = encode(Cmd.PING, arg=0)
    stream = fake + good
    result = find_frame(stream)
    assert result is not None
    cmd, arg, end = result
    assert cmd == Cmd.PING
    assert end == len(stream)
