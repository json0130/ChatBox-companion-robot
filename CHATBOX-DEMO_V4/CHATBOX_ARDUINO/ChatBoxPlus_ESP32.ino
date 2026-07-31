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
enum RobotState { IDLE, LISTEN, EXECUTE, SLEEP };

RobotState currentState = IDLE;
String serialInput = "";
unsigned long sleepTimer = 0;

// An expression name typed into the USB Serial Monitor, picked up by loop().
// Set by updatePanTracking(), which routes non-numeric serial lines here so the
// robot can be driven from a laptop with no Jetson attached. Numeric lines are
// still pixel errors and go straight to the pan servo.
String pendingCommand = "";

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
  // Modem sleep is on by default and makes the ESP32 miss ARP requests between
  // DTIM beacons — it associates and reports an IP, but nothing can reach it and
  // the TCP server below never sees a connection. We are USB powered, so there
  // is nothing to gain from sleeping.
  WiFi.setSleep(false);

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

  // An expression name typed into the Serial Monitor runs the same EXECUTE path
  // as a TCP command — bench testing without the Jetson. It also wakes us from
  // SLEEP, which otherwise only listens on TCP.
  if (pendingCommand.length() > 0 && currentState != EXECUTE) {
    serialInput = pendingCommand;
    pendingCommand = "";
    if (checkValidity(serialInput) || checkSequenceValidity(serialInput)) {
      Serial.println("[Serial] Running: " + serialInput);
      currentState = EXECUTE;
    } else {
      Serial.println("[Serial] Unknown command: " + serialInput);
      currentState = IDLE;
      sleepTimer = millis();
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
        // Loop it: executeExpression() advances ONE step per call (it is gated
        // on the 900ms step timer), so a single call would play only the first
        // pose before the return-to-default below overwrote it. Same pattern as
        // executeSequence() and the default reset further down.
        while (executeExpression(serialInput));
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
