/*
 * ChatBoxPlus Control System for ESP32
 * Author: Peter Cheong (Modified for ESP32)
 *
 * Main file — handles state machine and WiFi TCP communication.
 * Commands arrive from the Jetson over a persistent TCP socket (port 8888).
 * ESP32 is discoverable as chatbox.local via mDNS.
 * USB Serial is kept as a debug output only.
 */

#include <ESP32Servo.h>
#include <WiFi.h>
#include <ESPmDNS.h>

// ==================== WiFi / TCP Config ==================== //
// Uncomment ONE of the two blocks below depending on your network.

// ── Option A: Normal home/router WiFi (WPA2-Personal) ──────
#define WIFI_SSID     "ChatBox-AP"
#define WIFI_PASSWORD "chatbox1234"

// ── Option B: University / eduroam WiFi (WPA2-Enterprise) ──
// To use, uncomment #define USE_WPA2_ENTERPRISE and fill in your
// university credentials. Comment out Option A's WIFI_SSID above.
//
// #define USE_WPA2_ENTERPRISE
// #define WIFI_SSID    "eduroam"          // or your uni's SSID
// #define EAP_IDENTITY "abc123@aucklanduni.ac.nz"
// #define EAP_PASSWORD "your_uni_password"

// esp_eap_client.h / esp_wpa2.h not needed — WiFi.begin() handles WPA2-Enterprise directly in core 3.x

#define TCP_PORT      8888

WiFiServer server(TCP_PORT);
WiFiClient tcpClient;

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
// Single pan servo that keeps the tracked face centred. The Jetson streams one
// integer per line over USB serial: the horizontal pixel error (cx - centre).
//    error > 0  -> face is RIGHT of centre -> turn head right
//    error < 0  -> face is LEFT  of centre -> turn head left
// Because the camera moves with the head this is a closed feedback loop: each
// error nudges the angle a little, driving the error toward zero.
//
// Pin 19 is deliberately outside the expression-servo set (32/33/5/4/16/17 body,
// 23/27/26/12/13 head) so the head can pan without touching a gesture servo.
//
// Gating: panning only happens outside the EXECUTE state, so a gesture and the
// head never move at once. The Jetson also holds tracking during speech and
// sends "0" keep-alives — this is the hardware-side half of the same contract.
#define PAN_PIN 19
#define PAN_CENTER 90
#define PAN_MIN 20             // travel limits (degrees)
#define PAN_MAX 160
#define PAN_KP 0.03f           // degrees of angle per pixel of error
#define PAN_DEADBAND 30        // ignore small errors (matches the Jetson side)
#define PAN_MAX_INPUT_LEN 12

Servo panServo;
float panAngle = PAN_CENTER;
String panInput = "";
unsigned long lastPanMsg = 0;

// ==================== Function Declarations ==================== //
void servoInit();
void panServoInit();
void updatePanTracking();
bool executeExpression(String expression);
bool checkValidity(String input);
bool checkSequenceValidity(String input);
int getSequenceIndex(String input);
void executeSequence(String seqName);

// ==================== State Variables ==================== //
enum RobotState { IDLE, LISTEN, EXECUTE, SLEEP };

RobotState currentState = IDLE;
String serialInput = "";
unsigned long sleepTimer = 0;

// ==================== Setup ==================== //
void setup() {
  Serial.begin(115200);
  Serial.println("ESP32 ChatBox System Starting...");
  Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());

  setCpuFrequencyMhz(80);
  servoInit();
  panServoInit();

  // ── WiFi ──────────────────────────────────────────────────
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);

#ifdef USE_WPA2_ENTERPRISE
  // WPA2-Enterprise (eduroam) — uses WiFi.begin() enterprise overload in ESP32 core 3.x
  // No extra headers needed; WPA2_AUTH_PEAP is built into WiFi.h
  WiFi.begin(WIFI_SSID, WPA2_AUTH_PEAP, EAP_IDENTITY, EAP_IDENTITY, EAP_PASSWORD);
  Serial.print("[WiFi] Connecting (WPA2-Enterprise)");
#else
  // Normal WPA2-Personal
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("[WiFi] Connecting");
#endif

  unsigned long wifiStart = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - wifiStart > 30000) {
      Serial.println("\n[WiFi] Timeout — check credentials or network");
      break;
    }
    delay(500);
    Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED)
    Serial.println("\n[WiFi] Connected: " + WiFi.localIP().toString());
  else
    Serial.println("[WiFi] Not connected — continuing without WiFi");

  // ── mDNS — advertise as chatbox.local ────────────────────
  if (MDNS.begin("chatbox"))
    Serial.println("[mDNS] chatbox.local ready");
  else
    Serial.println("[mDNS] failed to start");

  // ── TCP server ────────────────────────────────────────────
  server.begin();
  Serial.println("[TCP] Listening on port " + String(TCP_PORT));

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

  // Accept a new TCP client whenever the previous one drops
  if (!tcpClient || !tcpClient.connected()) {
    WiFiClient c = server.available();
    if (c) {
      tcpClient = c;
      Serial.println("[TCP] Client connected from " + tcpClient.remoteIP().toString());
    }
  }

  switch (currentState) {

    case SLEEP:
      if (tcpClient && tcpClient.available()) {
        currentState = LISTEN;
        Serial.println("ChatBox: Waking up -> LISTEN");
      }
      break;

    case IDLE:
      if (tcpClient && tcpClient.available()) {
        currentState = LISTEN;
        Serial.println("ChatBox: IDLE -> LISTEN");
      } else if (millis() - sleepTimer > 30000) {
        currentState = SLEEP;
        Serial.println("ChatBox: Going to sleep...");
        while (executeExpression("sleep"));
        Serial.println("ChatBox: IDLE -> SLEEP");
      }
      break;

    case LISTEN:
      serialInput = tcpClient.readStringUntil('\n');
      serialInput.trim();
      Serial.println("DEBUG: Received [" + serialInput + "] Length:" + String(serialInput.length()));

      if (checkValidity(serialInput) || checkSequenceValidity(serialInput)) {
        tcpClient.println("OK:" + serialInput);
        currentState = EXECUTE;
        Serial.println("ChatBox: Valid command -> EXECUTE");
      } else {
        tcpClient.println("ERR:" + serialInput);
        Serial.println("ChatBox: Invalid command!");
        currentState = IDLE;
        sleepTimer = millis();
      }
      break;

    case EXECUTE:
      Serial.printf("Free heap before %s: %d bytes\n", serialInput.c_str(), ESP.getFreeHeap());
      Serial.println("//===================== " + serialInput + " =====================//");

      if (checkSequenceValidity(serialInput)) {
        executeSequence(serialInput);
      }
      else if (checkValidity(serialInput)) {
        executeExpression(serialInput);
      }

      tcpClient.println("DONE:" + serialInput);
      Serial.printf("Free heap after: %d bytes\n", ESP.getFreeHeap());
      sleepTimer = millis();
      currentState = IDLE;

      Serial.println("//===================== returning to default =====================//");
      while (executeExpression("default"));
      Serial.println("ChatBox: EXECUTE -> IDLE");
      break;
  }
}

// ==================== panServoInit ==================== //
void panServoInit() {
  panServo.attach(PAN_PIN, 500, 2400);
  panServo.write(PAN_CENTER);
  panAngle = PAN_CENTER;
  lastPanMsg = millis();
  Serial.println("[FaceTracking] pan servo ready on D" + String(PAN_PIN) +
                 " — streaming pixel error @ 115200");
}

// ==================== updatePanTracking ==================== //
// Reads integer pixel-error lines from USB Serial and eases the pan servo.
// Non-blocking: parses only what is already buffered, moves at most once per
// call. Expression commands arrive over TCP, so reading Serial here never
// competes with them; only complete all-integer lines are treated as errors.
void updatePanTracking() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      panInput.trim();
      if (panInput.length() > 0) {
        int error = panInput.toInt();
        lastPanMsg = millis();
        if (abs(error) > PAN_DEADBAND) {
          // error > 0 (face right) -> turn head right -> DECREASE angle.
          // If the head turns the wrong way, flip to '+= error * PAN_KP'.
          panAngle -= error * PAN_KP;
          panAngle = constrain(panAngle, PAN_MIN, PAN_MAX);
          panServo.write((int)(panAngle + 0.5f));
        }
      }
      panInput = "";
    } else if (panInput.length() < PAN_MAX_INPUT_LEN) {
      panInput += c;
    }
  }

  // No data for 2s: hold the current angle (uncomment to recentre instead).
  if (millis() - lastPanMsg > 2000) {
    // panServo.write(PAN_CENTER); panAngle = PAN_CENTER;
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
