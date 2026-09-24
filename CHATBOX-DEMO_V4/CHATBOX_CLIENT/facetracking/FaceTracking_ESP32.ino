/*
 * FaceTracking_ESP32.ino
 * BRING-UP SKETCH ONLY — not what runs on the robot.
 *
 * Use this to bench-test the pan servo and the serial link on their own,
 * without flashing the whole ChatBox system. The real implementation is the
 * pan block in CHATBOX_ARDUINO/ChatBoxPlus_ESP32.ino, which does the same job
 * alongside the expression servos and the TCP command socket.
 *
 * >> PIN DIFFERENCE: this sketch drives D13. The real firmware uses D19. <<
 *    Wire from ChatBoxPlus_ESP32.ino, not from here.
 *
 * The Jetson side is CHATBOX_CLIENT/OutputModules/face_tracking_output.py,
 * driven by robot.py. (It replaced the old standalone face_tracker_esp32.py.)
 * It streams a single integer per line over USB serial: the horizontal pixel
 * error (cx - center_x) of the tracked person.
 *    error > 0  ->  face is to the RIGHT of center  ->  turn head RIGHT
 *    error < 0  ->  face is to the LEFT  of center  ->  turn head LEFT
 *    error == 0 ->  keep-alive: subject seen and centered, hold this angle
 *
 * Because the camera moves with the head, this is a closed feedback loop:
 * each error nudges the servo angle a little, driving the error toward zero
 * (proportional control on the angle rather than an absolute pixel->angle map).
 *
 * NOTE: unlike the real firmware, this sketch recenters to PAN_CENTER after
 * LOST_TIMEOUT. That homing is deliberately disabled in ChatBoxPlus_ESP32.ino —
 * see the comment there before copying this behaviour across.
 */

#include <ESP32Servo.h>

// ==================== Servo Pin ==================== //
#define PAN_PIN 13             // D13 — single pan servo

// ==================== Servo Geometry ==================== //
#define PAN_CENTER 90          // servo angle that points straight ahead
#define PAN_MIN 20             // right/left travel limits (degrees)
#define PAN_MAX 160

// ==================== Control Tuning ==================== //
#define BAUD 115200
#define DEADBAND 30            // ignore small pixel errors (matches the Jetson side)
#define KP 0.04f               // proportional gain: angle step per pixel of error
#define MAX_STEP 4.0f          // clamp per-command motion (deg) so it never jerks
#define SMOOTH_STEP 1          // deg per tick when easing the servo toward target
#define UPDATE_INTERVAL 15     // ms between servo eases




#define LOST_TIMEOUT 3000      // ms without a command -> slowly recenter the head

Servo pan;

float panAngle = PAN_CENTER;   // current commanded servo angle (deg)
float panTarget = PAN_CENTER;  // where we want the servo to be (deg)
unsigned long lastUpdate = 0;
unsigned long lastCommand = 0;

String serialInput = "";

// ==================== Helpers ==================== //
// Apply a new pixel error from the Jetson to the pan target.
void applyError(int error) {
  lastCommand = millis();

  if (abs(error) <= DEADBAND) return;  // within tolerance, hold position

  // error > 0 (face right) -> turn head right -> DECREASE servo angle.
  // If the head turns the WRONG way, flip this sign (use +KP).
  float step = -KP * error;
  if (step >  MAX_STEP) step =  MAX_STEP;
  if (step < -MAX_STEP) step = -MAX_STEP;

  panTarget += step;
  if (panTarget > PAN_MAX) panTarget = PAN_MAX;
  if (panTarget < PAN_MIN) panTarget = PAN_MIN;
}

// Ease the physical servo toward panTarget so motion stays smooth.
void easeToTarget() {
  if (panAngle < panTarget) {
    panAngle += SMOOTH_STEP;
    if (panAngle > panTarget) panAngle = panTarget;
  } else if (panAngle > panTarget) {
    panAngle -= SMOOTH_STEP;
    if (panAngle < panTarget) panAngle = panTarget;
  }
  pan.write((int)(panAngle + 0.5f));
}

// ==================== Setup ==================== //
void setup() {
  Serial.begin(BAUD);
  delay(200);

  // ESP32Servo timer + pulse-width setup
  ESP32PWM::allocateTimer(0);
  pan.setPeriodHertz(50);
  pan.attach(PAN_PIN, 500, 2400);

  pan.write(PAN_CENTER);  // center the head
  lastCommand = millis();

  Serial.println("[FaceTracking] Ready. Single pan servo on D13, streaming pixel error @ 115200.");
}

// ==================== Main Loop ==================== //
void loop() {
  // ---- Read integer pixel-error lines from the Jetson ----
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      serialInput.trim();
      if (serialInput.length() > 0) {
        applyError(serialInput.toInt());
      }
      serialInput = "";
    } else if (serialInput.length() < 12) {
      serialInput += c;
    }
  }

  unsigned long now = millis();

  // ---- Recenter slowly if we have not seen a face for a while ----
  if (now - lastCommand > LOST_TIMEOUT) {
    panTarget = PAN_CENTER;
  }

  // ---- Ease the servo toward the target at a fixed rate ----
  if (now - lastUpdate >= UPDATE_INTERVAL) {
    lastUpdate = now;
    easeToTarget();
  }
}
