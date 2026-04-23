// grasp_controller.ino — receive single-byte serial commands, drive 5 servos via PCA9685.
//
// Protocol (matches pipeline/src/bci_grasp/deployment/arduino_serial.py):
//   '1' (0x31)  GRASP  → sweep servos toward SERVO_MAX
//   '0' (0x30)  REST   → sweep servos toward SERVO_MIN
//   any other byte is ignored (resilient to startup noise + stray chars in a
//   serial monitor).
//
// Control flow: non-blocking incremental stepping.
//
//   Why step-and-wait rather than commanding the target in one go:
//     Hobby servos have no speed input — the PWM pulse only encodes a
//     position, and the servo's built-in controller drives there at its
//     full physical speed (~0.1 s / 60°). Sending the final target in one
//     write means a jerky snap, hard on the linkages and drawing a large
//     current spike. To control the speed we instead feed a sequence of
//     small intermediate targets, each close enough that the servo reaches
//     it within SERVO_STEP_DELAY_MS before the next one arrives. Effective
//     speed = SERVO_STEP / SERVO_STEP_DELAY_MS; at the defaults (10 ticks
//     per 5 ms) a full MIN↔MAX traversal takes ~225 ms — smooth motion you
//     can watch, with no mechanical slam.
//
//   Each loop() iteration advances every servo one SERVO_STEP tick toward
//   its target (chosen by `c`). Servos that have reached the target no-op
//   — no retrigger, no snap-back. Because the loop isn't trapped inside a
//   blocking sweep, checkSerial() fires every 5 ms and `c` can redirect a
//   motion in flight.
//
// Hardware:
//   PCA9685 16-channel PWM driver @ 50 Hz (standard hobby-servo rate).
//   Servos wired to channels 0..NUM_SERVOS-1.
//   IMPORTANT: drive the PCA9685 V+ rail from a separate 5 V supply (bench
//   PSU or battery) — 5 servos at stall can pull >2 A, well beyond Arduino
//   USB limits. The Arduino 5V pin should power only the PCA9685 VCC (logic).
//
// Requires the "Adafruit PWM Servo Driver Library" from the Arduino Library
// Manager (Sketch → Include Library → Manage Libraries → search "Adafruit PWM").

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Default I2C address (0x40). Change only if you've jumpered the address pads.
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// ---- Servo constants (reused from reference_motor_control.cpp) --------------
// Tick values at 50 Hz on the PCA9685.
//   150 ≈ 0.5 ms pulse (full counter-clockwise on most hobby servos)
//   600 ≈ 2.5 ms pulse (full clockwise)
// Step/delay pace matches the reference sketch: 10 ticks per step, 5 ms
// between steps → ~225 ms for a full MIN↔MAX traversal.
static const uint8_t  NUM_SERVOS          = 5;
static const uint16_t SERVO_MIN           = 150;   // rest-side extreme
static const uint16_t SERVO_MAX           = 600;   // grasp-side extreme
static const uint16_t SERVO_STEP          = 10;    // PWM ticks per iteration
static const uint16_t SERVO_STEP_DELAY_MS = 5;     // delay per iteration

// State variable. Set by the most recent valid serial byte; loop() picks
// which direction each servo moves based on its value.
static char c = '0';  // start in REST

// Where each servo currently is. Updated in lockstep with the PCA9685 writes
// so the next loop() iteration knows where to step from. Initialized in
// setup() to match the boot pose.
static uint16_t current_pose[NUM_SERVOS];


void setup() {
    Serial.begin(115200);
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);

    // PCA9685 init. pwm.begin() calls Wire.begin() internally, so I2C is
    // ready after this returns.
    pwm.begin();
    pwm.setPWMFreq(50);  // 50 Hz = standard hobby-servo rate (20 ms period)

    // Park every servo at the rest extreme on boot so the hand starts in a
    // known pose rather than wherever the servos last stopped. Boot snaps
    // (no sweep) because there's no prior position to interpolate from.
    for (uint8_t i = 0; i < NUM_SERVOS; i++) {
        current_pose[i] = SERVO_MIN;
        pwm.setPWM(i, 0, current_pose[i]);
    }
}

void checkSerial() {
    // Peek for a new byte. If nothing is waiting, available() returns 0 and we
    // skip the read entirely — so the stepping below is NOT gated by serial
    // traffic. If a byte IS waiting and it's a valid command, it updates `c`
    // which redirects the next step.
    if (Serial.available() > 0) {
        char incoming = (char)Serial.read();
        if (incoming == '0' || incoming == '1') {
            c = incoming;
            Serial.println(c);  // debug echo
        }
        // any other byte is ignored
    }
}

void loop() {
    // ---- 1. Non-blocking serial check ---------------------------------------
    checkSerial();

    // ---- 2. Pick target based on current state ------------------------------
    // Both targets are global extremes today. If you ever want per-finger
    // tuning, swap target for a uint16_t target[NUM_SERVOS] array indexed
    // inside the loop below.
    const uint16_t target = (c == '1') ? SERVO_MAX : SERVO_MIN;

    // ---- 3. Step every servo ONE tick toward the target ---------------------
    // When a servo reaches the target, the if/else-if ladder no-ops for it —
    // no command is re-sent, no snap-back. All servos move in lockstep today
    // because they share one target, but this code handles divergent targets
    // correctly too (useful later for per-finger tuning).
    bool any_moved = false;
    for (uint8_t i = 0; i < NUM_SERVOS; i++) {
        if (current_pose[i] < target) {
            uint16_t next = current_pose[i] + SERVO_STEP;
            if (next > target) next = target;  // clamp to avoid overshoot
            current_pose[i] = next;
            pwm.setPWM(i, 0, current_pose[i]);
            any_moved = true;
        } else if (current_pose[i] > target) {
            uint16_t delta = current_pose[i] - target;
            current_pose[i] -= (delta < SERVO_STEP) ? delta : SERVO_STEP;
            pwm.setPWM(i, 0, current_pose[i]);
            any_moved = true;
        }
        // else: already at target — no write, no movement.
    }

    // ---- 4. Pace + LED feedback --------------------------------------------
    // Only delay if we actually moved. When the hand is holding a pose, the
    // loop runs at full speed and responds to the next serial byte
    // immediately — no 5 ms lag on state changes.
    if (any_moved) {
        delay(SERVO_STEP_DELAY_MS);
    }

    // LED reflects COMMANDED state (c), not the physical position. Lights up
    // the moment '1' arrives, clears the moment '0' arrives, regardless of
    // where mid-sweep the servos are.
    digitalWrite(LED_BUILTIN, (c == '1') ? HIGH : LOW);
}
