"""Manual Arduino-link tester: PsychoPy GUI with a grasp/rest toggle + stop button.

Purpose: exercise the Python→Arduino serial bridge without needing the
classifier. Click the big button to flip between REST and GRASP; watch the
Arduino LED (or servos, once wired) respond. Click STOP to quit.

Mirrors the PsychoPy conventions in ``reference/simple_bci/task.py``:
  - Global refresh-rate constant (monitor-dependent, CRITICAL FOR TIMING).
  - Frame-based main loop (one iteration per ``win.flip()``).
  - Clean teardown on exit so the port and window always close.

Usage
-----
    conda activate psychopy_env
    python scripts/test_arduino_gui.py --port /dev/cu.usbmodem1101
    python scripts/test_arduino_gui.py --port /dev/cu.usbmodem1101 --baud 115200
    python scripts/test_arduino_gui.py --port MOCK   # dry-run without real hardware

With ``--port MOCK`` no serial port is opened; the GUI prints what it
would send to stdout. Useful for developing the GUI on a machine without
the Arduino plugged in.
"""

from __future__ import annotations

import argparse
import logging
import sys

import psychopy.event
import psychopy.visual

from bci_grasp.config import load_config
from bci_grasp.deployment.arduino_serial import ArduinoLink

log = logging.getLogger("test_arduino_gui")

# ------------------------------------------------------------------
# CRITICAL FOR TIMING
# If this doesn't match your monitor, the GUI still works (there is no
# paradigm timing here) but frame pacing will feel off. Verify once with
# win.getActualFrameRate() if you care.
# ------------------------------------------------------------------
REFRESH_RATE_HZ = 60.0


# --- Mock transport for --port MOCK -----------------------------------------


class MockLink:
    """Stand-in for ArduinoLink that prints instead of writing to serial.

    Exposes the same ``send`` / context-manager interface so the GUI code
    doesn't branch on mock vs real. Useful for iterating on the GUI on a
    machine without the Arduino plugged in.
    """

    def __init__(self) -> None:
        self._last_state: str | None = None

    def __enter__(self) -> "MockLink":
        print("[MockLink] opened (no real serial port)")
        return self

    def __exit__(self, *args) -> None:
        print("[MockLink] closed")

    def send(self, state: str) -> None:
        byte = "1" if state == "grasp" else "0"
        print(f"[MockLink] would send {byte!r} (state={state})")
        self._last_state = state

    @property
    def is_open(self) -> bool:
        return True

    @property
    def last_state(self) -> str | None:
        return self._last_state


# --- Button helper ----------------------------------------------------------


class Button:
    """A clickable rectangle + centered text label.

    Rolling our own rather than using psychopy.visual.ButtonStim keeps this
    portable across PsychoPy versions (ButtonStim was revamped and its API
    is not stable).

    Parameters
    ----------
    win : psychopy.visual.Window
    pos : (x, y) in pixel units (window is created with units='pix').
    size : (w, h) in pixels.
    label : initial text shown in the button.
    fill : fill color in PsychoPy's [-1, 1] RGB.
    label_color : text color.
    """

    def __init__(self, win, pos, size, label, fill, label_color=(1, 1, 1)):
        self._win = win
        self._pos = pos
        self._size = size
        self._rect = psychopy.visual.Rect(
            win,
            width=size[0],
            height=size[1],
            pos=pos,
            fillColor=fill,
            lineColor=(1, 1, 1),
            lineWidth=2,
            units="pix",
        )
        self._text = psychopy.visual.TextStim(
            win,
            text=label,
            pos=pos,
            color=label_color,
            height=int(size[1] * 0.3),
            units="pix",
            alignText="center",
        )

    def draw(self) -> None:
        self._rect.draw()
        self._text.draw()

    def set_label(self, label: str) -> None:
        self._text.text = label

    def set_fill(self, color) -> None:
        self._rect.fillColor = color

    def contains(self, xy) -> bool:
        """True if point ``xy`` (in pixel units) is inside this button."""
        x, y = xy
        px, py = self._pos
        w, h = self._size
        return (px - w / 2) <= x <= (px + w / 2) and (py - h / 2) <= y <= (py + h / 2)


# --- Main loop --------------------------------------------------------------


def run_gui(link) -> None:
    """Open the PsychoPy window, run the click loop until STOP or Esc.

    The caller is responsible for the link's lifecycle (this function
    assumes ``link`` is already entered — i.e. port is open).
    """
    win = psychopy.visual.Window(
        size=[800, 500],
        fullscr=False,
        color=(-0.6, -0.6, -0.6),  # dark grey
        units="pix",
        allowGUI=True,
    )

    # Colors for the toggle button in each state.
    REST_FILL = (-0.3, -0.3, 0.6)   # bluish
    GRASP_FILL = (0.6, -0.2, -0.2)  # reddish

    toggle = Button(
        win, pos=(0, 50), size=(400, 180),
        label="REST — click for GRASP",
        fill=REST_FILL,
    )
    stop = Button(
        win, pos=(0, -160), size=(200, 70),
        label="STOP",
        fill=(-0.1, -0.1, -0.1),
    )
    status = psychopy.visual.TextStim(
        win, text="",
        pos=(0, -240), color=(1, 1, 1), height=18, units="pix",
        alignText="center",
    )

    mouse = psychopy.event.Mouse(win=win)
    current_state = "rest"
    # Send the initial state once so the Arduino matches the GUI from frame 0.
    link.send(current_state)

    prev_click = False  # mouse-edge detector (ignore held clicks)

    def refresh_ui() -> None:
        """Update labels/colors from current_state + status line."""
        if current_state == "rest":
            toggle.set_label("REST  —  click for GRASP")
            toggle.set_fill(REST_FILL)
        else:
            toggle.set_label("GRASP  —  click for REST")
            toggle.set_fill(GRASP_FILL)
        port_info = getattr(link, "port", "MOCK")
        status.text = (
            f"state: {current_state.upper()}    "
            f"last sent: '{getattr(link, 'last_state', None) or '-'}'    "
            f"port: {port_info}"
        )

    refresh_ui()

    running = True
    while running:
        # Esc as an emergency quit.
        if psychopy.event.getKeys(keyList=["escape"]):
            running = False
            break

        mx, my = mouse.getPos()
        is_down = any(mouse.getPressed())

        # Rising-edge: only act when the mouse goes from up → down this frame.
        if is_down and not prev_click:
            if toggle.contains((mx, my)):
                current_state = "grasp" if current_state == "rest" else "rest"
                link.send(current_state)
                log.info("toggled → %s", current_state)
                refresh_ui()
            elif stop.contains((mx, my)):
                log.info("STOP pressed")
                running = False
                break

        prev_click = is_down

        toggle.draw()
        stop.draw()
        status.draw()
        win.flip()

    win.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        help="Serial device (e.g. /dev/cu.usbmodem1101). Default: configs/deployment.yaml. "
        "Use --port MOCK to dry-run without hardware.",
    )
    parser.add_argument("--baud", type=int, help="Baud rate. Default: configs/deployment.yaml.")
    args = parser.parse_args()

    cfg = load_config("deployment")["serial"]
    port = args.port or cfg["port"]
    baud = args.baud or cfg["baud"]

    if port == "MOCK":
        with MockLink() as link:
            run_gui(link)
        return

    try:
        with ArduinoLink(
            port=port,
            baud=baud,
            open_delay_s=cfg["open_delay_s"],
            write_timeout_s=cfg["write_timeout_s"],
        ) as link:
            log.info("opened %s @ %d baud (waited %.1fs for Arduino boot)",
                     port, baud, cfg["open_delay_s"])
            run_gui(link)
    except Exception as e:
        log.error("serial error: %s", e)
        log.error("tip: pass --port MOCK to test the GUI without the Arduino plugged in.")
        sys.exit(1)


if __name__ == "__main__":
    main()
