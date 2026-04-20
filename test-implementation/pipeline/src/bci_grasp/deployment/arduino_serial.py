"""Serial transport for the prosthetic arm.

Thin ``pyserial`` wrapper around the binary frame protocol defined in
``protocol.py``. Keeps the transport concerns (port open/close, timeouts,
Arduino reset settling) out of the inference loop.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from . import protocol

if TYPE_CHECKING:
    import serial


class ArduinoLink:
    """Context-manager wrapper over a serial port speaking the Cmd protocol.

    Usage
    -----
        with ArduinoLink("/dev/tty.usbmodem101") as link:
            link.send(protocol.Cmd.GRASP)
            link.send(protocol.Cmd.REST)
    """

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        open_delay_s: float = 2.0,
        write_timeout_s: float = 0.2,
    ) -> None:
        """Remember parameters; the port is opened lazily in ``__enter__``."""
        self.port = port
        self.baud = baud
        self.open_delay_s = open_delay_s
        self.write_timeout_s = write_timeout_s
        self._ser: "serial.Serial | None" = None

    def __enter__(self) -> "ArduinoLink":
        """Open the port and wait for the Arduino to finish its DTR-triggered reset."""
        raise NotImplementedError(
            "Implement: self._ser = serial.Serial(self.port, self.baud, "
            "write_timeout=self.write_timeout_s); time.sleep(self.open_delay_s); "
            "return self."
        )

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the port."""
        raise NotImplementedError("Implement: self._ser.close().")

    def send(self, cmd: protocol.Cmd | int, arg: int = 0) -> None:
        """Encode and write a single command frame."""
        raise NotImplementedError(
            "Implement: self._ser.write(protocol.encode(cmd, arg)); self._ser.flush()."
        )
