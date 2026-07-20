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

// Static IP removed — DHCP reservation on Jetson (dnsmasq) guarantees
// MAC bc:dd:c2:cc:a6:34 always gets 10.42.0.100. DHCP is more reliable
// because it keeps ARP active; static IP on ESP32 causes ARP to go silent.

WiFiServer tcpServer(TCP_PORT);
WiFiClient tcpClient;
bool wifiOk = false;  // set true once WiFi connects; TCP features gated on this

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

// ==================== Function Declarations ==================== //
void servoInit();
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
bool commandFromSerial = false;  // true when command arrived over USB serial, false for TCP

// ==================== Response Helper ==================== //
// Routes OK/ERR/DONE responses back to whichever transport sent the command.
// Debug prints always go to Serial regardless.
void sendResponse(String msg) {
  if (commandFromSerial) {
    Serial.println(msg);
  } else if (tcpClient && tcpClient.connected()) {
    tcpClient.println(msg);
  }
}

// ==================== Setup ==================== //
void setup() {
  Serial.begin(115200);
  delay(10);

  // ── WiFi (optional — Serial commands work without it) ──────────────────
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("[+] Connecting to %s (15s timeout) ", WIFI_SSID);

  unsigned long wifiStart = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 15000) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    wifiOk = true;
    Serial.println("\n[+] WiFi Connected: " + WiFi.localIP().toString());
    tcpServer.begin();
    Serial.printf("[+] TCP Server on port %d\n", TCP_PORT);
  } else {
    wifiOk = false;
    Serial.println("\n[!] WiFi unavailable — Serial-only mode. TCP disabled.");
    Serial.println("[!] Start ChatBox-AP hotspot and reset to enable TCP.");
  }
  // ───────────────────────────────────────────────────────────────────────

  servoInit();
  sleepTimer = millis();
}

// ==================== Main Loop ==================== //
void loop() {
  // ── WiFi watchdog — background reconnect, never blocks or restarts ──────
  if (wifiOk && WiFi.status() != WL_CONNECTED) {
    wifiOk = false;
    Serial.println("[WiFi] Lost — attempting background reconnect (Serial still active)");
    WiFi.reconnect();
  }
  if (!wifiOk && WiFi.status() == WL_CONNECTED) {
    wifiOk = true;
    Serial.println("[WiFi] Reconnected: " + WiFi.localIP().toString());
    tcpServer.begin();
  }

  // Accept a new TCP client whenever the previous one drops (WiFi must be up)
  if (wifiOk && (!tcpClient || !tcpClient.connected())) {
    WiFiClient c = tcpServer.available();
    if (c) {
      tcpClient = c;
      sleepTimer = millis();
      Serial.println("[TCP] Client connected from " + tcpClient.remoteIP().toString());
    }
  }

  switch (currentState) {

    case SLEEP:
      if (tcpClient && tcpClient.available()) {
        commandFromSerial = false;
        currentState = LISTEN;
        Serial.println("ChatBox: Waking up (TCP) -> LISTEN");
      } else if (Serial.available()) {
        commandFromSerial = true;
        currentState = LISTEN;
        Serial.println("ChatBox: Waking up (Serial) -> LISTEN");
      }
      break;

    case IDLE:
      if (tcpClient && tcpClient.available()) {
        commandFromSerial = false;
        currentState = LISTEN;
        Serial.println("ChatBox: IDLE -> LISTEN (TCP)");
      } else if (Serial.available()) {
        commandFromSerial = true;
        sleepTimer = millis();  // serial activity counts as "connected" for sleep timer
        currentState = LISTEN;
        Serial.println("ChatBox: IDLE -> LISTEN (Serial)");
      } else if ((!wifiOk || !tcpClient.connected()) && millis() - sleepTimer > 30000) {
        // Only sleep when Jetson is not connected via either transport
        currentState = SLEEP;
        Serial.println("ChatBox: Going to sleep...");
        while (executeExpression("sleep"));
        Serial.println("ChatBox: IDLE -> SLEEP");
      }
      break;

    case LISTEN:
      if (commandFromSerial) {
        serialInput = Serial.readStringUntil('\n');
      } else {
        serialInput = tcpClient.readStringUntil('\n');
      }
      serialInput.trim();
      Serial.println("DEBUG: Received [" + serialInput + "] via " + (commandFromSerial ? "Serial" : "TCP"));

      if (checkValidity(serialInput) || checkSequenceValidity(serialInput)) {
        sendResponse("OK:" + serialInput);
        currentState = EXECUTE;
        Serial.println("ChatBox: Valid command -> EXECUTE");
      } else {
        sendResponse("ERR:" + serialInput);
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

      sendResponse("DONE:" + serialInput);
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
