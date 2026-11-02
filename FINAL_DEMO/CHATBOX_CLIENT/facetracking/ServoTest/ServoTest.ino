/*
 * ServoTest.ino
 * Dead-simple bring-up test for the single pan servo on D13.
 *
 * No Jetson, no serial input needed. Flash this, and the servo should
 * continuously sweep left <-> right. If it DOES sweep here but not in
 * FaceTracking_ESP32.ino, the problem is serial/parsing, not the hardware.
 * If it does NOT sweep here, the problem is power, wiring, or the pin.
 *
 * Wiring reminder:
 *   - Servo signal (usually orange/white) -> GPIO 13 (D13)
 *   - Servo V+  (red)   -> EXTERNAL 5V supply  (NOT the ESP32 3V3 pin)
 *   - Servo GND (brown/black) -> supply GND  AND  ESP32 GND (common ground!)
 */

#include <ESP32Servo.h>

#define PAN_PIN 13

Servo pan;

void setup() {
  Serial.begin(115200);
  delay(200);

  ESP32PWM::allocateTimer(0);
  pan.setPeriodHertz(50);
  pan.attach(PAN_PIN, 500, 2400);

  Serial.println("[ServoTest] Sweeping servo on D13...");
}

void loop() {
  // Sweep 20 -> 160 degrees
  for (int a = 20; a <= 160; a += 2) {
    pan.write(a);
    Serial.printf("angle = %d\n", a);
    delay(20);
  }
  // Sweep back 160 -> 20
  for (int a = 160; a >= 20; a -= 2) {
    pan.write(a);
    Serial.printf("angle = %d\n", a);
    delay(20);
  }
}
