// ==================================================================== //
// StyleControl.ino — affective styling for the expression servos.
//
// Drop this file into the sketch folder alongside ChatBoxPlus_ESP32.ino and
// ServoControl.ino, then make the four small edits listed in INTEGRATION.md.
// Nothing in here runs until those edits are made, so adding the file alone
// changes no behaviour.
//
// What it does
// ------------
// The stock firmware plays every gesture identically: a tag arrives, a fixed
// five-step move set runs, the same angles every time for every persona in every
// mood. This adds five values that reshape that same sequence without replacing
// any of it:
//
//   amplitude  how much of the authored travel to perform   0.30 .. 1.00
//   droop      signed valence tint, sad <-> bright         -1.00 .. 1.00
//   posture    standing offset on neck and shoulders       -1.00 .. 1.00
//   tempo      playback speed                               0.50 .. 1.60
//   idle       how often to stir between gestures           0.05 .. 1.00
//
// `droop` is the one that answers "make the greeting look sad". Amplitude cannot
// do that — a smaller wave is still a happy wave. Valence needs a signed offset
// that pushes ears, brows, eyelids, neck and shoulders down when the mood is
// unpleasant and lifts them when it is pleasant, on top of whatever tag plays.
//
// Safety
// ------
// Every styled angle is clamped to the span the stock move sets already reach,
// so styling can never command a pose the unstyled firmware would not have
// commanded. Amplitude therefore tops out at 1.00: each servo has only three
// symbols, so D and U already *are* the ends of its range and asking for more
// would only saturate. The authored gestures are full expression; styling damps.
// ==================================================================== //

// The constants and prototypes live in the header so they are in scope in
// ServoControl.ino too — see the note at the top of StyleControl.h for why that
// matters with the IDE's file ordering.
#include "StyleControl.h"

// ── Style state ─────────────────────────────────────────────────────── //
// Defaults are the identity: until a STYLE message arrives the robot behaves
// exactly as it does today.
float styleAmplitude = 1.0f;
float styleDroop     = 0.0f;
float stylePosture   = 0.0f;
float styleTempo     = 1.0f;
float styleIdle      = 0.45f;

// Negative means "use styleAmplitude". Raised to 1.0 for home poses — see the
// note in StyleControl.h.
float styleAmpOverride = -1.0f;

// Poses that exist to reach a position rather than to express something.
bool styleIsHomePose(const String &tag) {
  return tag == "default" || tag == "sleep";
}

// Called at the start of every gesture, before any setX() runs.
void styleBeginGesture(const String &tag) {
  styleAmpOverride = styleIsHomePose(tag) ? 1.0f : -1.0f;
}

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

static inline uint8_t clampAngle(float v, int lo, int hi) {
  if (v < lo) v = lo;
  if (v > hi) v = hi;
  return (uint8_t)(v + 0.5f);
}

// ── Applying the style ──────────────────────────────────────────────── //
//
// One servo, one step. `target` is the angle the stock firmware would have
// written; `rest` is that servo's M pose, which amplitude scales away from;
// `lo`/`hi` are the ends of the range the stock move sets already use.
//
// Called from the setX() functions after they have computed their angle, so the
// existing symbol tables stay the single source of truth for the poses.
uint8_t styleAngle(float target, int rest, int lo, int hi,
                   int droopDeg, int postureDeg) {
  float amp = (styleAmpOverride >= 0.0f) ? styleAmpOverride : styleAmplitude;
  float a = rest + amp * (target - rest);
  a += styleDroop * droopDeg;
  a += stylePosture * postureDeg;
  return clampAngle(a, lo, hi);
}

// Step duration for the gesture player. Tempo divides the base interval, so
// higher tempo means a brisker performance of the same angles.
unsigned long styleStepMs(unsigned long baseMs) {
  return (unsigned long)(baseMs / clampf(styleTempo, STYLE_TEMPO_MIN,
                                         STYLE_TEMPO_MAX));
}

// Idle interval: 0.05 -> roughly every 20 s, 1.0 -> roughly every second.
unsigned long styleIdleMs() {
  float f = clampf(styleIdle, 0.05f, 1.0f);
  return (unsigned long)(1000.0f / f);
}

// ── Parsing "STYLE a t p d i" ───────────────────────────────────────── //
//
// Returns true if the line was a style command and was consumed. Anything else
// is left alone for the existing expression handling, so the tag protocol is
// untouched. Values are clamped on arrival — never trust the wire, a stray digit
// should not be able to drive a servo into its end stop.
bool handleStyleCommand(const String &line) {
  if (!line.startsWith("STYLE")) return false;

  float v[5];
  int found = 0;
  int i = 5;                       // just past "STYLE"
  while (found < 5 && i < (int)line.length()) {
    while (i < (int)line.length() && line[i] == ' ') i++;
    int start = i;
    while (i < (int)line.length() && line[i] != ' ') i++;
    if (i > start) v[found++] = line.substring(start, i).toFloat();
  }
  if (found != 5) {
    Serial.println("[Style] malformed, expected: STYLE amp tempo posture droop idle");
    return true;                   // consumed, but ignored
  }

  styleAmplitude = clampf(v[0], STYLE_AMP_MIN, STYLE_AMP_MAX);
  styleTempo     = clampf(v[1], STYLE_TEMPO_MIN, STYLE_TEMPO_MAX);
  stylePosture   = clampf(v[2], -1.0f, 1.0f);
  styleDroop     = clampf(v[3], -1.0f, 1.0f);
  styleIdle      = clampf(v[4], 0.05f, 1.0f);

  Serial.print("[Style] amp "); Serial.print(styleAmplitude);
  Serial.print("  tempo ");     Serial.print(styleTempo);
  Serial.print("  posture ");   Serial.print(stylePosture);
  Serial.print("  droop ");     Serial.print(styleDroop);
  Serial.print("  idle ");      Serial.println(styleIdle);
  return true;
}

void styleReset() {
  styleAmplitude = 1.0f;
  styleDroop = 0.0f;
  stylePosture = 0.0f;
  styleTempo = 1.0f;
  styleIdle = 0.45f;
  Serial.println("[Style] reset to neutral");
}
