"""Minimal serial link to the prosthetic-arm Arduino.

v1 protocol: one ASCII byte per state transition.

    Python → Arduino
    ----------------
    b'1'   GRASP   → move servos to SERVO_MAX
    b'0'   REST    → move servos to SERVO_MIN
    other          → ignored by the Arduino sketch (robust to startup noise)

Why so simple: the arm has two states. A framed protocol with CRC / sync /
multi-byte commands is reserved for when we add a 3rd command (speed,
calibration, telemetry back). See ``deployment/protocol.py`` for the framed
version — kept as reference, not wired here.

The ``ArduinoLink`` class is a context manager so the port always closes
cleanly, even on exception. Usage::

    with ArduinoLink("/dev/cu.usbmodem1101") as link:
        link.send("grasp")
        ...
        link.send("rest")
"""

from __future__ import annotations

import time
from typing import Literal

import serial

State = Literal["grasp", "rest"]

GRASP_BYTE = b"1"
REST_BYTE = b"0"


class ArduinoLink:
    """Thin wrapper around ``pyserial.Serial`` that speaks the single-byte protocol.

    Lazy-open: the port is not opened until ``__enter__`` so the class can be
    constructed cheaply (and mocked in tests).
    """

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        open_delay_s: float = 2.0,
        write_timeout_s: float = 0.2,
    ) -> None:
        self.port = port
        self.baud = baud
        self.open_delay_s = open_delay_s
        self.write_timeout_s = write_timeout_s
        self._ser: serial.Serial | None = None
        self._last_state: State | None = None

    def __enter__(self) -> "ArduinoLink":
        """Open the port, then block for open_delay_s to let the Arduino finish booting."""
        self._ser = serial.Serial(
            self.port,
            self.baud,
            write_timeout=self.write_timeout_s,
        )
        # Arduinos auto-reset on DTR drop (port-open triggers it). Sketch
        # isn't running yet; anything written here would be lost.
        time.sleep(self.open_delay_s)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            finally:
                self._ser = None

    def send(self, state: State) -> None:
        """Write the byte corresponding to ``state``.

        Parameters
        ----------
        state : {"grasp", "rest"}
            Desired arm state. Raises ``ValueError`` on anything else.

        Notes
        -----
        Sends unconditionally — does not deduplicate repeats. Callers that
        stream classifier output should apply their own "send on transition"
        logic (see ``configs/deployment.yaml::min_dwell_s``) to avoid
        flooding the serial line and wearing out the servos.
        """
        if self._ser is None:
            raise RuntimeError(
                "ArduinoLink not open. Use as a context manager: `with ArduinoLink(...) as link:`"
            )
        if state == "grasp":
            byte = GRASP_BYTE
        elif state == "rest":
            byte = REST_BYTE
        else:
            raise ValueError(f"state must be 'grasp' or 'rest', got {state!r}")

        self._ser.write(byte)
        self._ser.flush()
        self._last_state = state

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    @property
    def last_state(self) -> State | None:
        """Most recent state successfully sent, or None if nothing sent yet."""
        return self._last_state
