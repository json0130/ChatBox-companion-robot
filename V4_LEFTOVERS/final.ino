/*
 * ChatBoxPlus Control System for ESP32
 * Author: Peter Cheong (Modified for ESP32)
 *
 * Main file — handles state machine and wired USB Serial communication.
 * The ESP32 is plugged into the Jetson by USB cable; no WiFi is used.
 * One serial line carries both kinds of message from the Jetson:
 *   - integers ("-12", "0") are face-tracking pixel errors -> pan servo
 *   - anything else is an expression/sequence name -> EXECUTE
 * Replies to commands are "OK:<cmd>", "ERR:<cmd>" and "DONE:<cmd>" lines;
 * every other line printed here is debug output and is ignored by the Jetson.
 */

#include <ESP32Servo.h>

// ==================== Constants ==================== //
#define NUM_VALID_EXPRESSIONS 22
#define MAX_SEQUENCE_LENGTH 20  // max steps per sequence

String validExpressions[] = {
  "greeting", "wave", "point", "confused", "shrug", "angry", "sad", "sleep",
  "default", "pose", "idle", "dance_sway", "dance_arms", "dance_groove",
  "hands_clap", "hands_wave_both", "head_nod", "head_shake",
  "ears_wiggle", "ears_perk", "hands_only_wave", "hands_only_tap"
};

// ==================== Sequence Definitions ==================== //
// Add or edit your sequences here — just list expression names in order.
// End every sequence with ""  (empty string acts as terminator)

String seq_dance[] = {
  "default", "hands_only_tap", "hands_wave_both",  "dance_groove",
  "dance_sway", "hands_clap", "hands_wave_both", "dance_sway",
  "ears_wiggle", "hands_only_wave", "ears_wiggle", "pose", ""
};

// ==================== Sequence Registry ==================== //
// Add your sequence name + pointer here — keep both arrays in sync!
#define NUM_SEQUENCES 1

String sequenceNames[] = {
  "seq_dance"
};

String* sequences[] = {
  seq_dance
};


// ==================== Face-Tracking Pan Servo ==================== //
// The pan servo lives in ServoControl.ino alongside the expression servos —
// see panServoInit() and updatePanTracking() there for the pin, the tuning
// constants, and the pixel-error protocol the Jetson speaks.
//
// Gating: panning only happens outside the EXECUTE state, so a gesture and the
// head never move at once. The Jetson also holds tracking during speech and
// sends "0" keep-alives — this is the hardware-side half of the same contract.

// ==================== Function Declarations ==================== //
void servoInit();
void panServoInit();
void updatePanTracking();
bool isIntegerLine(const String &s);
bool executeExpression(String expression);
bool checkValidity(String input);
bool checkSequenceValidity(String input);
int getSequenceIndex(String input);
void executeSequence(String seqName);

// ==================== State Variables ==================== //
enum RobotState { IDLE, EXECUTE, SLEEP };

RobotState currentState = IDLE;
String serialInput = "";
unsigned long sleepTimer = 0;

// An expression name received over USB Serial (from the Jetson, or typed into
// the Serial Monitor), picked up by loop(). Set by updatePanTracking(), which
// routes non-numeric serial lines here. Numeric lines are pixel errors and go
// straight to the pan servo.
String pendingCommand = "";

// ==================== Setup ==================== //
void setup() {
  // Room for the face-tracking stream that piles up while a gesture blocks loop().
  Serial.setRxBufferSize(1024);
  Serial.begin(115200);
  Serial.println("ESP32 ChatBox System Starting...");
  Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());

  setCpuFrequencyMhz(80);
  servoInit();
  panServoInit();

  Serial.println("\nValid expressions:");
  for (int i = 0; i < NUM_VALID_EXPRESSIONS; i++) {
    Serial.print(validExpressions[i]);
    if (i < NUM_VALID_EXPRESSIONS - 1) Serial.print(", ");
  }

  Serial.println("\nValid sequences:");
  for (int i = 0; i < NUM_SEQUENCES; i++) {
    Serial.print(sequenceNames[i]);
    if (i < NUM_SEQUENCES - 1) Serial.print(", ");
  }
  Serial.println();
}

// ==================== Main Loop ==================== //
void loop() {
  // Face tracking runs every loop EXCEPT during EXECUTE (a gesture is playing).
  // EXECUTE blocks inside the switch below, so control only reaches here between
  // commands — panning is naturally paused while a gesture runs.
  updatePanTracking();

  // A command from the Jetson (or the Serial Monitor) — also wakes us from SLEEP.
  if (pendingCommand.length() > 0 && currentState != EXECUTE) {
    serialInput = pendingCommand;
    pendingCommand = "";
    Serial.println("DEBUG: Received [" + serialInput + "] Length:" + String(serialInput.length()));
    if (checkValidity(serialInput) || checkSequenceValidity(serialInput)) {
      Serial.println("OK:" + serialInput);
      currentState = EXECUTE;
      Serial.println("ChatBox: Valid command -> EXECUTE");
    } else {
      Serial.println("ERR:" + serialInput);
      Serial.println("ChatBox: Invalid command!");
      currentState = IDLE;
      sleepTimer = millis();
    }
  }

  switch (currentState) {

    case SLEEP:
      // Woken by the pendingCommand block above.
      break;

    case IDLE:
      if (millis() - sleepTimer > 30000) {
        currentState = SLEEP;
        Serial.println("ChatBox: Going to sleep...");
        while (executeExpression("sleep"));
        Serial.println("ChatBox: IDLE -> SLEEP");
      }
      break;

    case EXECUTE:
      Serial.printf("Free heap before %s: %d bytes\n", serialInput.c_str(), ESP.getFreeHeap());
      Serial.println("//===================== " + serialInput + " =====================//");

      if (checkSequenceValidity(serialInput)) {
        executeSequence(serialInput);
      }
      else if (checkValidity(serialInput)) {
        // Loop it: executeExpression() advances ONE step per call (it is gated
        // on the 900ms step timer), so a single call would play only the first
        // pose before the return-to-default below overwrote it. Same pattern as
        // executeSequence() and the default reset further down.
        while (executeExpression(serialInput));
      }

      Serial.println("DONE:" + serialInput);
      Serial.printf("Free heap after: %d bytes\n", ESP.getFreeHeap());
      sleepTimer = millis();
      currentState = IDLE;

      Serial.println("//===================== returning to default =====================//");
      while (executeExpression("default"));
      Serial.println("ChatBox: EXECUTE -> IDLE");
      break;
  }
}

// ==================== executeSequence ==================== //
void executeSequence(String seqName) {
  int idx = getSequenceIndex(seqName);
  if (idx == -1) {
    Serial.println("Sequence not found: " + seqName);
    return;
  }

  String* seq = sequences[idx];
  int step = 0;

  while (step < MAX_SEQUENCE_LENGTH) {
    String expr = seq[step];
    if (expr == "") break;  // terminator reached

    Serial.println("  -> Step " + String(step + 1) + ": " + expr);
    while (executeExpression(expr));  // run expression to completion

    step++;
  }

  Serial.println("Sequence complete: " + seqName);
}

// ==================== getSequenceIndex ==================== //
int getSequenceIndex(String input) {
  input.trim();
  input.toLowerCase();
  for (int i = 0; i < NUM_SEQUENCES; i++) {
    if (input == sequenceNames[i]) return i;
  }
  return -1;
}

// ==================== checkSequenceValidity ==================== //
bool checkSequenceValidity(String input) {
  return getSequenceIndex(input) != -1;
}
