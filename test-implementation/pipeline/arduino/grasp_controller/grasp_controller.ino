// grasp_controller.ino — receive framed serial commands, drive 5 servos via PCA9685.
//
// Protocol (matches pipeline/src/bci_grasp/deployment/protocol.py):
//   5-byte frame: [SYNC0=0xAA][SYNC1=0x55][CMD][ARG][CRC8]
//   CRC-8 polynomial 0x07, init 0x00, no reflection, no final XOR,
//   computed over CMD + ARG (2 bytes).
//
// Commands:
//   0x01 GRASP      → move all servos to SERVO_MAX (closed-hand pose)
//   0x02 REST       → move all servos to SERVO_MIN (open-hand rest pose)
//   0x03 PING       → no-op, used to verify the link is alive
//   0x04 CALIBRATE  → sweep once through the full range (debug aid)
//
// TODO: this is a scaffold. The state machine below receives + validates
// frames but only blinks the onboard LED on each command. Wire up the
// PCA9685 servo drive once hardware is on the desk.

#include <Arduino.h>
// #include <Wire.h>
// #include <Adafruit_PWMServoDriver.h>

// ---- Protocol constants (mirror protocol.py) --------------------------------
static const uint8_t SYNC0 = 0xAA;
static const uint8_t SYNC1 = 0x55;
static const size_t  FRAME_LEN = 5;

enum Cmd : uint8_t {
    CMD_GRASP     = 0x01,
    CMD_REST      = 0x02,
    CMD_PING      = 0x03,
    CMD_CALIBRATE = 0x04,
};

// ---- Servo constants (reused from reference_motor_control.cpp) --------------
static const uint8_t NUM_SERVOS = 5;
static const uint16_t SERVO_MIN = 150;  // PCA9685 tick count, 50 Hz PWM
static const uint16_t SERVO_MAX = 600;

// Adafruit_PWMServoDriver pwm;  // uncomment once wiring is done

// ---- Receive ring buffer ----------------------------------------------------
static uint8_t rx_buf[FRAME_LEN];
static size_t  rx_len = 0;

// ---- CRC-8 (poly 0x07, init 0x00, no reflection, no final XOR) -------------
static uint8_t crc8(const uint8_t *data, size_t len) {
    uint8_t crc = 0x00;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (uint8_t b = 0; b < 8; b++) {
            if (crc & 0x80) crc = (uint8_t)((crc << 1) ^ 0x07);
            else            crc = (uint8_t)(crc << 1);
        }
    }
    return crc;
}

// ---- Command handlers -------------------------------------------------------
static void handle_grasp(uint8_t arg) {
    (void)arg;
    digitalWrite(LED_BUILTIN, HIGH);
    // TODO: for (i=0..NUM_SERVOS) pwm.setPWM(i, 0, SERVO_MAX);
}

static void handle_rest(uint8_t arg) {
    (void)arg;
    digitalWrite(LED_BUILTIN, LOW);
    // TODO: for (i=0..NUM_SERVOS) pwm.setPWM(i, 0, SERVO_MIN);
}

static void handle_ping(uint8_t arg) {
    (void)arg;
    // No-op. Could optionally echo a PING frame back over Serial to
    // close a round-trip latency check.
}

static void handle_calibrate(uint8_t arg) {
    (void)arg;
    // TODO: sweep SERVO_MIN → SERVO_MAX → SERVO_MIN once.
}

// Dispatch a validated frame.
static void dispatch(uint8_t cmd, uint8_t arg) {
    switch (cmd) {
        case CMD_GRASP:     handle_grasp(arg);     break;
        case CMD_REST:      handle_rest(arg);      break;
        case CMD_PING:      handle_ping(arg);      break;
        case CMD_CALIBRATE: handle_calibrate(arg); break;
        default: break;  // already filtered in try_decode(); here for safety
    }
}

// Attempt to decode the current rx_buf as a frame. Shift out on failure so
// we resynchronize to the next possible SYNC0.
//
// Returns true if a valid frame was dispatched.
static bool try_decode() {
    if (rx_len < FRAME_LEN) return false;

    if (rx_buf[0] != SYNC0 || rx_buf[1] != SYNC1) {
        // Shift left by 1 and drop the oldest byte — look for sync later.
        for (size_t i = 1; i < rx_len; i++) rx_buf[i - 1] = rx_buf[i];
        rx_len--;
        return false;
    }

    uint8_t cmd = rx_buf[2];
    uint8_t arg = rx_buf[3];
    uint8_t got = rx_buf[4];
    uint8_t want = crc8(&rx_buf[2], 2);

    if (got != want) {
        // Bad CRC: discard just the first sync byte so a real frame that
        // happened to start inside this noise window still has a chance.
        for (size_t i = 1; i < rx_len; i++) rx_buf[i - 1] = rx_buf[i];
        rx_len--;
        return false;
    }

    // Filter unknown commands the same way — drop the sync byte, keep looking.
    if (cmd != CMD_GRASP && cmd != CMD_REST && cmd != CMD_PING && cmd != CMD_CALIBRATE) {
        for (size_t i = 1; i < rx_len; i++) rx_buf[i - 1] = rx_buf[i];
        rx_len--;
        return false;
    }

    dispatch(cmd, arg);
    rx_len = 0;  // frame consumed
    return true;
}

void setup() {
    Serial.begin(115200);
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);

    // TODO (once hardware is ready):
    // pwm.begin();
    // pwm.setPWMFreq(50);
    // for (uint8_t i = 0; i < NUM_SERVOS; i++) pwm.setPWM(i, 0, SERVO_MIN);
}

void loop() {
    while (Serial.available() > 0 && rx_len < FRAME_LEN) {
        rx_buf[rx_len++] = (uint8_t)Serial.read();
    }
    // Keep draining until no more frames are decodable from the current buffer.
    while (try_decode()) { /* loop */ }
}
