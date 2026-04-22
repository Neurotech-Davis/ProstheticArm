# Real_Time_Signal

Real-time EEG classification → serial command to motor hardware.

## Files

- `real_time.py` — original. Classifies eyes-open vs eyes-closed (alpha band), fires a single `b'1'` to a **Spresense** board when a closed-eyes streak is detected. One-shot trigger with cooldown.
- `real_time_arduino.py` — adapted variant targeting an **Arduino** with a bidirectional state machine.
- `Test.py` — scratch / sanity script.

## real_time_arduino.py

Inspired by `real_time.py`. Same data path (OpenBCI Cyton via BrainFlow → Welch PSD → alpha-band power → scaler + LDA). Different output protocol and control logic.

### Changes from `real_time.py`

| Aspect | `real_time.py` | `real_time_arduino.py` |
|---|---|---|
| Target device | Spresense | Arduino |
| Serial var | `SERIAL_PORT`, `spresense` | `SERIAL_PORT_ARDUINO`, `arduino` |
| Output | One-shot `b'1'` on closed-eyes detection | `b'1'` on open→closed, `b'0'` on closed→open |
| Control | Single streak + cooldown | Bidirectional state machine with hysteresis |
| Thresholds | Single `CONFIDENCE_THRESHOLD` | Asymmetric `ENTER_CLOSED_THRESHOLD` + `ENTER_OPEN_THRESHOLD` (dead zone between) |
| Startup | Silent | Sends `b'0'` to force Arduino to known rest pose |

Feature extraction, artifact rejection, window size, and BrainFlow setup are unchanged.

### How it works

1. **Stream**: BrainFlow pulls a 2 s window from the Cyton board each tick.
2. **Extract**: per channel → center, 58–62 Hz bandstop, Welch PSD, mean power in 8–13 Hz (alpha).
3. **Classify**: scaler + LDA → `P(eyes_closed)`.
4. **Zone check**:
   - `conf ≥ 0.80` → increment closed-streak, reset open-streak
   - `conf ≤ 0.40` → increment open-streak, reset closed-streak
   - otherwise → dead zone, reset both
5. **Transition (gated by state + streak + cooldown)**:
   - `open` and closed-streak ≥ 3 and cooldown elapsed → send `b'1'`, state = `closed`
   - `closed` and open-streak ≥ 3 and cooldown elapsed → send `b'0'`, state = `open`
6. **Artifact window** (rail or filter fail): both streaks reset, state unchanged.

### Key features

- **Asymmetric hysteresis**: entry into each state requires the classifier in that zone; dead zone between thresholds prevents chatter near the decision boundary.
- **Streak gate**: a transition only fires after `REQUIRED_STREAK = 3` consecutive in-zone windows.
- **Cooldown**: after any transition, `COOLDOWN_SEC = 3.0` must elapse before the next can fire. Prevents servo thrash.
- **Startup sync**: initial `b'0'` aligns Arduino pose with Python state on boot.
- **Heartbeat log**: every tick prints state, confidence, and both streak counters so you can see what the system is about to do.

### Arduino firmware requirement

Must read ASCII bytes over serial at 115200 baud:
- `b'0'` → move servos to rest
- `b'1'` → move servos to grasp
- anything else → ignore

### Tunables (top of file)

- `ENTER_CLOSED_THRESHOLD`, `ENTER_OPEN_THRESHOLD` — hysteresis band
- `REQUIRED_STREAK` — persistence before firing
- `COOLDOWN_SEC` — refractory lock after transitions
- `WINDOW_SIZE_SEC`, `AMPLITUDE_THRESHOLD` — signal window + artifact gate
- `SERIAL_PORT_ARDUINO`, `SERIAL_PORT_BCI` — update to match your machine
