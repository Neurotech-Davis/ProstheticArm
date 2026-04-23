#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm;

#define NUM_SERVOS 5
#define SERVO_MIN 150
#define SERVO_MAX 600
#define SERVO_DELAY 1000

int servoPins[NUM_SERVOS] = {0, 1, 2, 3, 4};

void setup() {
  Serial.begin(9600);
  Serial.println("Alternate Servo Test");

  pwm.begin();
  pwm.setPWMFreq(50);

  // initialize to default positions
  for (int i = 0; i < NUM_SERVOS; i++) {
    pwm.setPWM(servoPins[i], 0, SERVO_MIN);
  }
}

void loop() {
  // Rotate all servos to one extreme position
  for (int pos = SERVO_MIN; pos <= SERVO_MAX; pos += 10) {
    for (int i = 0; i < NUM_SERVOS; i++) {
      pwm.setPWM(servoPins[i], 0, pos);
    } 
    delay(5);
  }

  delay(SERVO_DELAY);

  // Rotate all servos back to their minimum position
  for (int pos = SERVO_MAX; pos >= SERVO_MIN; pos -= 10) {
    for (int i = 0; i < NUM_SERVOS; i++) {
      pwm.setPWM(servoPins[i], 0, pos);
    }
    delay(5);
  }

  delay(SERVO_DELAY);
}
