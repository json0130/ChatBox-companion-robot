// ==================================================================== //
// StyleControl.h — tuning constants and prototypes for the styling layer.
//
// These live in a header rather than in StyleControl.ino for a concrete reason:
// the Arduino IDE concatenates the .ino files in a sketch *alphabetically* after
// the main sketch, so ServoControl.ino is compiled before StyleControl.ino. A
// #define is textual — it has to appear before the line that uses it — so any
// weight defined in the .ino would be "not declared in this scope" at every call
// site in ServoControl.ino.
//
// A header sidesteps the ordering entirely: both .ino files #include this, so the
// constants are in scope wherever they are needed, whatever the file names are.
// (Which matters — renaming the sketch files changes the concatenation order.)
// ==================================================================== //

#ifndef STYLE_CONTROL_H
#define STYLE_CONTROL_H

#include <Arduino.h>

// ── Style clamps ─────────────────────────────────────────────────────── //
// Amplitude stops at 1.00 deliberately: each servo has only three symbols, so D
// and U already are the ends of its range and asking for more only saturates
// against the clamp. The authored gestures are full expression; styling damps.
#define STYLE_AMP_MIN 0.30f
#define STYLE_AMP_MAX 1.00f
#define STYLE_TEMPO_MIN 0.50f
#define STYLE_TEMPO_MAX 1.60f

// ── Droop and posture weights ────────────────────────────────────────── //
// Degrees at full deflection. Signed so positive droop reads as sadder and
// positive posture as more open. Ears carry the most because they are the
// highest-impact expressive channel on this build; hands carry none, because a
// drooping hand reads as a broken servo rather than a mood.
//
// Left-hand servos take the negated value at the call site — both sides sit at
// 90 +/- an offset, so a mood has to move them oppositely or the robot ends up
// lopsided.
//
// These are the tuning knobs. Start here if a mood does not read right, and keep
// them in step with DROOP_DEG / POSTURE_DEG in ../../servo_style.py or preview.py
// stops predicting what the robot actually does.
#define DROOP_EARS      -20
#define DROOP_BROW       12   // R adds, L subtracts
#define DROOP_EYELID    -10   // R subtracts toward its D pose
#define DROOP_NECK       -8
#define DROOP_SHOULDER  -12
#define POSTURE_NECK      10
#define POSTURE_SHOULDER   8

// ── Style state, defined in StyleControl.ino ─────────────────────────── //
// Defaults are the identity, so until a STYLE message arrives the robot behaves
// exactly as the unstyled firmware does.
extern float styleAmplitude;
extern float styleDroop;
extern float stylePosture;
extern float styleTempo;
extern float styleIdle;

// Set to 1.0 while a home pose plays, otherwise negative. Amplitude scales a
// servo toward its rest angle, which on a pose whose job is to *be* a position
// undoes it — 'default' puts the arms down at 50 degrees and an amplitude of 0.34
// drags them back up to 102. Droop and posture still apply, so the robot keeps
// looking sad while it waits instead of snapping back to a bright neutral.
extern float styleAmpOverride;

// True while a home pose is playing. setNeck() uses it to return the head level:
// droop and posture both push the neck the same way for a bright, dominant robot
// and together consume 12 of its 30 degrees, so the head would park visibly off
// centre and stay there. A slumped shoulder reads as mood; a permanently tilted
// head reads as a fault. Gestures are unaffected.
extern bool styleHomePose;

// ── Prototypes ───────────────────────────────────────────────────────── //
uint8_t styleAngle(float target, int rest, int lo, int hi,
                   int droopDeg, int postureDeg);
bool styleIsHomePose(const String &tag);
void styleBeginGesture(const String &tag);
unsigned long styleStepMs(unsigned long baseMs);
unsigned long styleIdleMs();
bool handleStyleCommand(const String &line);
void styleReset();

#endif  // STYLE_CONTROL_H
