#include <Arduino.h>

// Sony Spresense has 4 built-in LEDs. We will use the first one.
const int ledPin = PIN_LED0; 
char receivedChar;

void setup() {
  // Start serial communication at 115200 baud rate. 
  // (High baud rates are good for BCI applications to reduce latency)
  Serial.begin(115200);
  
  // Configure the LED pin as an output
  pinMode(ledPin, OUTPUT);
  
  // Ensure the LED is off to start
  digitalWrite(ledPin, LOW);
}

void loop() {
  // Check if any data has been sent from the Python script
  if (Serial.available() > 0) {
    
    // Read the incoming byte
    receivedChar = Serial.read();
    
    // Logic to toggle states (Open vs. Close for your future prosthetic)
    if (receivedChar == '1') {
      digitalWrite(ledPin, HIGH); // Turn LED ON (Simulates "Close hand")
      Serial.println("Hand Closed"); // Optional: Send a confirmation back to Python
    } 
    else if (receivedChar == '0') {
      digitalWrite(ledPin, LOW);  // Turn LED OFF (Simulates "Open hand")
      Serial.println("Hand Opened");
    }
  }
}