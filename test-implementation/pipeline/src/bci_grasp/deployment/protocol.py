"""Binary framed serial protocol: Python <-> Arduino.

Frame layout (5 bytes, fixed length, little-endian is N/A — all fields are uint8):

    +--------+--------+--------+--------+--------+
    | SYNC0  | SYNC1  |  CMD   |  ARG   |  CRC8  |
    | 0xAA   | 0x55   | uint8  | uint8  | uint8  |
    +--------+--------+--------+--------+--------+

- SYNC0/SYNC1 are constant magic bytes. Two-byte sync is standard on noisy
  serial links because a single resync byte collides with CMD/ARG values.
- CMD: one of the ``Cmd`` enum values below.
- ARG: command-specific 8-bit argument (e.g. grasp strength 0–255; reserved
  for now, set to 0).
- CRC8: polynomial 0x07 (ITU/CRC-8), init 0x00, no reflection, no final XOR.
  Computed over CMD and ARG only (2 bytes) — matches the Arduino reference
  implementations most Adafruit examples use.

Why binary and not ASCII:
  - Fixed length → the Arduino parser is a 5-byte ring buffer, zero
    heap allocation, ~dozens of lines total.
  - Sync bytes let the Arduino recover from mid-stream connection without
    waiting for a newline that might never arrive.
  - CRC catches the one-off bit flips you get on cheap USB cables.
"""

from __future__ import annotations

from enum import IntEnum

SYNC0 = 0xAA
SYNC1 = 0x55
FRAME_LEN = 5


class Cmd(IntEnum):
    """Command opcodes. Keep in sync with configs/deployment.yaml and the .ino."""

    GRASP = 0x01
    REST = 0x02
    PING = 0x03
    CALIBRATE = 0x04


def crc8(data: bytes) -> int:
    """CRC-8 (poly 0x07, init 0x00, no reflection, no final XOR).

    Standard ITU CRC-8. Kept in pure Python so there's no native dep — the
    messages are 2 bytes long, performance doesn't matter.
    """
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def encode(cmd: Cmd | int, arg: int = 0) -> bytes:
    """Build a 5-byte frame.

    Parameters
    ----------
    cmd : Cmd or int
        Command opcode (see Cmd enum).
    arg : int
        8-bit argument. Must be in ``[0, 255]``.

    Returns
    -------
    bytes
        Exactly 5 bytes: SYNC0, SYNC1, CMD, ARG, CRC8.
    """
    cmd_int = int(cmd)
    if not 0 <= cmd_int <= 0xFF:
        raise ValueError(f"cmd out of uint8 range: {cmd_int}")
    if not 0 <= arg <= 0xFF:
        raise ValueError(f"arg out of uint8 range: {arg}")
    payload = bytes([cmd_int, arg])
    return bytes([SYNC0, SYNC1, cmd_int, arg, crc8(payload)])


class FrameError(ValueError):
    """Raised when a received frame is malformed (sync / length / CRC)."""


def decode(frame: bytes) -> tuple[Cmd, int]:
    """Parse a 5-byte frame into ``(Cmd, arg)``.

    Raises
    ------
    FrameError
        If the frame is the wrong length, has bad sync bytes, a bad CRC,
        or carries an unrecognized CMD.

    Returns
    -------
    (Cmd, int)
        Validated command + 8-bit arg.

    Notes
    -----
    Intentionally strict: the caller is expected to have already synchronized
    to the SYNC0/SYNC1 boundary. For stream-level resync, see ``find_frame``.
    """
    if len(frame) != FRAME_LEN:
        raise FrameError(f"expected {FRAME_LEN} bytes, got {len(frame)}")
    if frame[0] != SYNC0 or frame[1] != SYNC1:
        raise FrameError(
            f"bad sync bytes: {frame[0]:#04x} {frame[1]:#04x} "
            f"(expected {SYNC0:#04x} {SYNC1:#04x})"
        )
    cmd_int, arg, got_crc = frame[2], frame[3], frame[4]
    want_crc = crc8(bytes([cmd_int, arg]))
    if got_crc != want_crc:
        raise FrameError(f"bad CRC: got {got_crc:#04x}, want {want_crc:#04x}")
    try:
        cmd = Cmd(cmd_int)
    except ValueError as e:
        raise FrameError(f"unknown CMD: {cmd_int:#04x}") from e
    return cmd, arg


def find_frame(stream: bytes) -> tuple[Cmd, int, int] | None:
    """Scan ``stream`` for the first valid frame.

    Used when resynchronizing after noise. Scans for the SYNC0/SYNC1 pair and
    tries to decode the next 3 bytes as CMD/ARG/CRC.

    Returns
    -------
    (Cmd, arg, end_index) or None
        ``end_index`` is one past the last byte of the consumed frame, so the
        caller can do ``stream = stream[end_index:]`` to advance past it.
        Returns None if no valid frame is found in ``stream``.
    """
    for i in range(len(stream) - FRAME_LEN + 1):
        if stream[i] == SYNC0 and stream[i + 1] == SYNC1:
            try:
                decode(stream[i : i + FRAME_LEN])
            except FrameError:
                continue
            cmd, arg = decode(stream[i : i + FRAME_LEN])
            return cmd, arg, i + FRAME_LEN
    return None
