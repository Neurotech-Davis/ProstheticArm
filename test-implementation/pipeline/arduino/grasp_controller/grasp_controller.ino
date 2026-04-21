// grasp_controller.ino — receive single-byte serial commands, drive servos via PCA9685.
//
// Protocol (matches pipeline/src/bci_grasp/deployment/arduino_serial.py):
//   '1' (0x31)  GRASP  → all servos to SERVO_MAX
//   '0' (0x30)  REST   → all servos to SERVO_MIN
//   any other byte is ignored (resilient to startup noise + stray chars in
//   a serial monitor).
//
// PCA9685 servo drive is wired up but commented out until hardware is on
// the desk — the sketch blinks LED_BUILTIN on each command so you can
// confirm the link works with just the USB cable.

#include <Arduino.h>
// #include <Wire.h>
// #include <Adafruit_PWMServoDriver.h>

// Adafruit_PWMServoDriver pwm;

// ---- Servo constants (reused from reference_motor_control.cpp) --------------
static const uint8_t  NUM_SERVOS = 5;
static const uint16_t SERVO_MIN  = 150;  // PCA9685 tick count, 50 Hz PWM → rest pose
static const uint16_t SERVO_MAX  = 600;  //                                → grasp pose

// Track current state so we don't re-command the same pose every byte.
static char current_state = '0';  // start in REST

// ---- Command handlers -------------------------------------------------------
static void handle_grasp() {
    digitalWrite(LED_BUILTIN, HIGH);
    // TODO (hardware ready): for (uint8_t i=0; i<NUM_SERVOS; i++) pwm.setPWM(i, 0, SERVO_MAX);
}

static void handle_rest() {
    digitalWrite(LED_BUILTIN, LOW);
    // TODO (hardware ready): for (uint8_t i=0; i<NUM_SERVOS; i++) pwm.setPWM(i, 0, SERVO_MIN);
}

void setup() {
    Serial.begin(115200);
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);

    // Hardware init (enable when PCA9685 is wired):
    // pwm.begin();
    // pwm.setPWMFreq(50);
    // for (uint8_t i = 0; i < NUM_SERVOS; i++) pwm.setPWM(i, 0, SERVO_MIN);
}

void loop() {
    while (Serial.available() > 0) {
        char c = (char)Serial.read();
        if (c != '0' && c != '1') continue;  // ignore anything else

        if (c == current_state) continue;    // no-op on duplicate
        current_state = c;

        if (c == '1') handle_grasp();
        else          handle_rest();
    }
}
