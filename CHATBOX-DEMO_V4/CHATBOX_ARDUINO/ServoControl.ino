/*
 * ServoControl.ino
 * This file contains the functions for controlling the servos of the ChatBoxPlus robot.
 * The functions in this file are used to set the positions of the servos based on the input commands.
 * 
 * Note: 
 * The command naming convention is as follows:
 * - R and L are used to indicate the right and left side of the robot respectively.
 * - H, M, and L are used to indicate high, medium, and low positions respectively.
 * - F, B, R, L, and _ are used to indicate front, back, right, left, and center positions respectively (only for neck). 
*/

// ==================== Constants and Definitions ==================== //
#define SIZE_OF_SET 5
#define D 0  // down
#define M 1  // middle
#define U 2  // up
#define R 3  // right
#define L 4  // left
#define C 6  // cancel/stop

// Body related servo pins
#define RNeck_Port 32
#define LNeck_Port 33
#define RShoulder_Port 5
#define LShoulder_Port 4
#define RHand_Port 16
#define LHand_Port 17

// Head related servo pins
#define Ears_Port 23
#define RBrow_Port 27
#define LBrow_Port 26
#define RELid_Port 12
#define LELid_Port 13

// Servo offset constants
#define HAND_OFFSET_DOWN -40
#define HAND_OFFSET_UP 60
#define SHOULDER_OFFSET_UP 80
#define SHOULDER_OFFSET_MIDDLE 50
#define SHOULDER_OFFSET_DOWN -40
#define EYE_OFFSET_UP 40
#define EYE_OFFSET_MIDDLE 20
#define BROW_OFFSET_MIDDLE 30
#define BROW_OFFSET_DOWN 60
#define EARS_UP 165
#define EARS_MIDDLE 130
#define EARS_DOWN 120

// NUM_VALID_EXPRESSIONS now defined in main file

// ==================== Servo Objects ==================== //
Servo RShoulder, LShoulder, RHand, LHand, RNeck, LNeck;
Servo Ears, RBrow, LBrow, RELid, LELid;

// Servo position variables
uint8_t RShoulderDest, LShoulderDest, RHandDest, LHandDest, RNeckDest, LNeckDest;
uint8_t EarsDest, RBrowDest, LBrowDest, RELidDest, LELidDest;

// Expression execution variables
int8_t count = 0;
unsigned long timer = 0;

// ==================== Move Set Structure ==================== //
struct moveSet {
  uint8_t EarsSet[SIZE_OF_SET];
  uint8_t RBrowSet;
  uint8_t LBrowSet;
  uint8_t REyeSet[SIZE_OF_SET];
  uint8_t LEyeSet[SIZE_OF_SET];
  uint8_t NeckSet[SIZE_OF_SET];
  uint8_t RShoulderSet[SIZE_OF_SET];
  uint8_t LShoulderSet[SIZE_OF_SET];
  uint8_t RHandSet[SIZE_OF_SET];
  uint8_t LHandSet[SIZE_OF_SET];
};

// ==================== Expression Definitions ==================== //
// struct moveSet greeting = {
//   { U, U, U, U, U }, { M }, { M },
//   { U, D, D, D, D }, { U, D, D, D, D }, { U, U, M, M, M },
//   { U, U, D, D, D }, { D, D, D, D, D },
//   { M, M, M, M, M }, { M, M, M, M, M }
// };

struct moveSet greeting = {
  { U, U, U, U, U }, { M }, { M },
  { U, D, D, D, D }, { U, D, D, D, D }, { M, M, M, M, M },
  { U, U, D, D, D }, { D, D, D, D, D },
  { M, M, M, M, M }, { M, M, M, M, M }
};

struct moveSet wave = {
  { U, U, U, U, U }, { M }, { M },
  { U, U, D, U, U }, { U, U, D, U, U }, { U, U, M, M, M },
  { U, U, U, U, U }, { D, D, D, D, D },
  { M, U, M, U, M }, { M, M, M, M, M }
};

struct moveSet point = {
  { U, U, U, U, U }, { U }, { M },  // R brow raised, L neutral (focused look)
  { U, U, U, U, U }, { U, U, U, U, U }, { M, R, R, R, M }, // neck turns slightly right
  { U, U, U, U, D }, { D, D, D, D, D },   // R shoulder lifts then drops back; L stays down
  { U, U, U, U, M }, { M, M, M, M, M }    // R hand extends forward (up) then returns
};

// struct moveSet confused = {
//   { U, U, U, U, U }, { U }, { D },
//   { U, D, U, D, U }, { U, D, U, D, U }, { M, R, R, R, R },
//   { M, M, M, M, M }, { M, M, M, M, M },
//   { M, M, U, U, U }, { M, M, U, U, U }
// };

// FIXED CONFUSED - Slower, less aggressive movements
struct moveSet confused = {
  { U, U, M, M, M }, // ears - removed rapid changes
  U, D,              // brows - kept different (confused look)
  { U, M, U, M, M }, // right eye - slower blinking
  { U, M, U, M, M }, // left eye - slower blinking  
  { M, R, M, L, M }, // neck - smoother head movement (not 3x R)
  { M, M, U, M, M }, // right shoulder - gentler movement
  { M, M, M, M, M }, // left shoulder - keep steady
  { M, M, M, M, M }, // right hand - keep steady
  { M, M, M, M, M }  // left hand - keep steady
};

struct moveSet shrug = {
  { M, M, M, M, M }, { U }, { U },
  { M, M, D, M, M }, { M, M, D, M, M }, { M, M, M, M, M },
  { U, U, U, U, U }, { U, U, U, U, U },
  { M, U, U, U, U }, { M, U, U, U, U }
};

// struct moveSet angry = {
//   { U, M, U, M, U }, { U }, { U },
//   { U, M, M, U, U }, { U, M, M, U, U }, { M, M, M, M, M },
//   { D, U, M, U, M }, { D, U, M, U, M },
//   { M, M, M, M, M }, { M, M, M, M, M }
// };

// FIXED ANGRY - Reduced simultaneous movements
struct moveSet angry = {
  { U, M, U, M, M }, // ears - less rapid flicking
  U, U,              // brows - both up (angry look)
  { U, M, U, M, M }, // right eye - slower changes
  { U, M, U, M, M }, // left eye - slower changes
  { M, M, M, M, M }, // neck - keep steady (focus anger in face)
  { M, U, M, M, M }, // right shoulder - one strong movement
  { M, M, U, M, M }, // left shoulder - offset timing
  { M, M, M, M, M }, // right hand - keep steady
  { M, M, M, M, M }  // left hand - keep steady
};

// IMPROVED SAD - More subtle, natural sadness
struct moveSet sad = {
  { M, M, D, D, D }, // ears - gradual droop (not sudden)
  M, M,              // brows - neutral (let eyes show sadness)
  { M, M, D, D, M }, // right eye - gentle downward look, return
  { M, M, D, D, M }, // left eye - gentle downward look, return
  { M, D, D, M, M }, // neck - slight downward head movement
  { M, M, D, M, M }, // right shoulder - subtle slump
  { M, M, D, M, M }, // left shoulder - subtle slump
  { M, M, M, M, M }, // right hand - keep steady
  { M, M, M, M, M }  // left hand - keep steady
};

struct moveSet sleepMode = {  // Fixed: was "sleepp"
  { D, D, D, D, C }, { M }, { M },
  { D, D, D, D, D }, { D, D, D, D, D }, { D, D, D, D, C },
  { D, D, D, D, C }, { D, D, D, D, C },
  { M, M, M, M, C }, { M, M, M, M, C }
};

struct moveSet defaultMode = {  // Fixed: was "defaultt"
  { U, U, U, U, C }, { M }, { M },
  { U, U, U, U, C }, { U, U, U, U, C }, { M, M, M, M, C },
  { D, D, D, D, C }, { D, D, D, D, C },
  { M, M, M, M, C }, { M, M, M, M, C }
};

struct moveSet pose = {
  { U, U, U, U, U }, { M }, { M },            // ears up, brows neutral
  { U, U, D, D, U }, { U, U, U, U, U },       // R eye blinks mid-sequence, L open
  { M, L, L, L, L },                           // neck turns to look at raised arm
  { D, M, U, U, D }, { D, D, D, D, D },       // RShoulder sweeps D→M→U→U→D; L stays down
  { D, M, U, U, D }, { M, M, M, M, M }        // RHand rises with shoulder D→M→U→U→D
};

// A more subtle idle_natural gesture with only ears and eyes moving.
struct moveSet idle= {
  { M, U, M, M, M }, // Ears: A quick twitch up and then back to neutral at the start.
  { M }, { M },             // Brows: Remain neutral.
  { M, M, D, M, M }, // Right Eye: A slow blink that happens in the middle of the sequence.
  { M, M, D, M, M }, // Left Eye: A slow blink that happens in the middle of the sequence.
  { M, M, M, M, M }, // Neck: Remains steady.
  { M, M, M, M, M }, // R Shoulder: Remains steady.
  { M, M, M, M, M }, // L Shoulder: Remains steady.
  { M, M, M, M, M }, // R Hand: Remains steady.
  { M, M, M, M, M }  // L Hand: NOW STEADY (no movement).
};

struct moveSet dance_sway = {
  { U, U, U, U, U },  // ears up throughout
  M, M,                // brows neutral
  { U, U, U, U, U },  // right eye open
  { U, U, U, U, U },  // left eye open
  { R, M, L, M, R },  // neck sways R → center → L → center → R
  { U, D, U, D, U },  // right shoulder pumps up/down
  { D, U, D, U, D },  // left shoulder offset (opposite phase)
  { M, U, M, M, M },  // right hand waves
  { M, M, M, M, M }   // left hand steady
};

struct moveSet dance_arms = {
  { U, U, M, U, U },  // ears pulse
  U, U,                // brows raised (excited look)
  { U, D, U, D, U },  // right eye blinks rhythmically
  { U, D, U, D, U },  // left eye blinks rhythmically
  { M, M, M, M, M },  // neck steady
  { U, M, D, M, U },  // right shoulder full sweep U→M→D→M→U
  { D, M, U, M, D },  // left shoulder opposite sweep
  { U, M, D, M, U },  // right hand follows shoulder
  { D, M, U, M, D }   // left hand follows left shoulder
};

struct moveSet dance_groove = {
  { U, M, U, M, U },  // ears bounce
  U, M,                // asymmetric brows (quirky look)
  { U, M, D, M, U },  // right eye full cycle
  { U, M, D, M, U },  // left eye full cycle
  { L, M, R, M, L },  // neck swings L → center → R → center → L
  { U, U, D, D, U },  // right shoulder hold high then drop
  { D, D, U, U, D },  // left shoulder opposite
  { U, M, M, D, U },  // right hand arcs
  { D, M, M, U, D }   // left hand counter-arc
};

struct moveSet hands_clap = {
  { M, M, M, M, M },  // ears steady
  M, M,
  { U, U, U, U, U },  // eyes open
  { U, U, U, U, U },
  { M, M, M, M, M },  // neck steady
  { U, U, U, U, U },  // both shoulders raise and hold
  { U, U, U, U, U },
  { U, D, U, D, M },  // right hand claps down/up alternating
  { D, U, D, U, M }   // left hand opposite phase
};

struct moveSet hands_wave_both = {
  { M, M, M, M, M },
  M, M,
  { U, U, U, U, U },
  { U, U, U, U, U },
  { M, M, M, M, M },  // neck steady
  { M, M, M, M, M },  // shoulders stay down
  { M, M, M, M, M },
  { U, M, U, M, M },  // right hand waves
  { M, U, M, U, M }   // left hand offset wave
};

struct moveSet head_nod = {
  { M, M, M, M, M },
  M, M,
  { U, U, U, U, U },
  { U, U, U, U, U },
  { U, D, U, D, M },  // neck nods up/down
  { M, M, M, M, M },  // everything else steady
  { M, M, M, M, M },
  { M, M, M, M, M },
  { M, M, M, M, M }
};

struct moveSet head_shake = {
  { M, M, M, M, M },
  M, M,
  { U, U, U, U, U },
  { U, U, U, U, U },
  { R, L, R, L, M },  // neck shakes R→L→R→L→center
  { M, M, M, M, M },
  { M, M, M, M, M },
  { M, M, M, M, M },
  { M, M, M, M, M }
};

struct moveSet ears_wiggle = {
  { U, D, U, D, M },  // ears wiggle up/down, return to middle
  M, M,
  { U, U, U, U, U },  // eyes open throughout
  { U, U, U, U, U },
  { M, M, M, M, M },  // everything else steady
  { M, M, M, M, M },
  { M, M, M, M, M },
  { M, M, M, M, M },
  { M, M, M, M, M }
};

struct moveSet ears_perk = {
  { M, U, U, U, M },  // ears slowly perk up and hold, return
  M, M,
  { U, U, U, U, U },
  { U, U, U, U, U },
  { M, M, M, M, M },
  { M, M, M, M, M },
  { M, M, M, M, M },
  { M, M, M, M, M },
  { M, M, M, M, M }
};

struct moveSet hands_only_wave = {
  { M, M, M, M, M },
  M, M,
  { U, U, U, U, U },
  { U, U, U, U, U },
  { M, M, M, M, M },
  { D, D, D, D, D },  // shoulders locked down so only wrists move
  { D, D, D, D, D },
  { U, M, U, M, M },  // right hand waves
  { M, U, M, U, M }   // left hand offset
};

struct moveSet hands_only_tap = {
  { M, M, M, M, M },
  M, M,
  { U, U, U, U, U },
  { U, U, U, U, U },
  { M, M, M, M, M },
  { D, D, D, D, D },  // shoulders locked down
  { D, D, D, D, D },
  { U, M, U, M, M },  // right hand taps
  { U, M, U, M, M }   // left hand same phase (tapping together)
};

struct moveSet listOfMoveSets[] = { 
  greeting, wave, point, confused, shrug, angry, sad, sleepMode, defaultMode, pose, idle, dance_sway, 
  dance_arms, dance_groove, hands_clap, hands_wave_both, head_nod, head_shake, 
  ears_wiggle, ears_perk, hands_only_wave, hands_only_tap 
};

// validExpressions now defined in main file

// ========================================== servoInit ========================================== //
void servoInit() {
  Serial.println("Initializing servos...");
  
  // Attach servos to pins
  RShoulder.attach(RShoulder_Port);
  RHand.attach(RHand_Port);
  LShoulder.attach(LShoulder_Port);
  LHand.attach(LHand_Port);
  RNeck.attach(RNeck_Port);
  LNeck.attach(LNeck_Port);
  
  Ears.attach(Ears_Port);
  RBrow.attach(RBrow_Port);
  LBrow.attach(LBrow_Port);
  RELid.attach(RELid_Port);
  LELid.attach(LELid_Port);

  // Set initial positions
  setNeck(M);
  setShoulder('R', M); setShoulder('L', M);
  setHand('R', M); setHand('L', M);
  setEars(M);
  setBrows('R', M); setBrows('L', M);
  setEyes('R', M); setEyes('L', M);
  updateServos();
  
  Serial.println("Servos initialized successfully!");
}

// ========================================== setHand ========================================== //
void setHand(char side, uint8_t value) {
  if (side != 'R' && side != 'L') {
    Serial.print("invalid Hand input");
    return;
  }

  int handOffset = (side == 'R') ? 1 : -1;

  switch (value) {
    case (D): handOffset *= HAND_OFFSET_DOWN; break;
    case (M): handOffset = 0; break;
    case (U): handOffset *= HAND_OFFSET_UP; break;
    default: Serial.println("Error: Invalid Hand value. Use D, M, U"); break;
  }

  if (side == 'R') {
    RHandDest = 90 + handOffset;
  } else {
    LHandDest = 90 + handOffset;
  }
}

// ========================================== setShoulder ========================================== //
void setShoulder(char side, uint8_t value) {
  if (side != 'R' && side != 'L') {
    Serial.print("invalid shoulder input");
    return;
  }

  int shoulderOffset = (side == 'R') ? 1 : -1;

  switch (value) {
    case (U): shoulderOffset *= SHOULDER_OFFSET_UP; break;
    case (M): shoulderOffset *= SHOULDER_OFFSET_MIDDLE; break;
    case (D): shoulderOffset *= SHOULDER_OFFSET_DOWN; break;
    default: Serial.println("Error: Invalid shoulder value. Use U, M, D"); break;
  }

  if (side == 'R') {
    RShoulderDest = 90 + shoulderOffset;
  } else {
    LShoulderDest = 90 + shoulderOffset;
  }
}

// ========================================== setNeck ========================================== //
void setNeck(uint8_t Orientation) {
  switch (Orientation) {
    case (D): RNeckDest = 70; LNeckDest = 110; break;
    case (U): RNeckDest = 100; LNeckDest = 80; break;
    case (R): RNeckDest = 75; LNeckDest = 85; break;
    case (L): RNeckDest = 100; LNeckDest = 120; break;
    case (M): RNeckDest = 82; LNeckDest = 103; break;
  }
}

// ========================================== setEyes ========================================== //
void setEyes(char side, uint8_t Position) {
  int eyeOffSet = (side == 'R') ? 1 : -1;
  switch (Position) {
    case (U): eyeOffSet *= EYE_OFFSET_UP; break;
    case (M): eyeOffSet *= EYE_OFFSET_MIDDLE; break;
    case (D): eyeOffSet *= 0; break;
  }

  if (side == 'R') {
    RELidDest = 90 + eyeOffSet;
  } else {
    LELidDest = 90 + eyeOffSet;
  }
}

// ======================================== setEars ========================================//
void setEars(uint8_t Position) {
  switch (Position) {
    case (U): EarsDest = EARS_UP; break;
    case (M): EarsDest = EARS_MIDDLE; break;
    case (D): EarsDest = EARS_DOWN; break;
    default: EarsDest = EARS_MIDDLE;
  }
}

// ======================================== setBrows ========================================//
void setBrows(char side, uint8_t position) {
  int8_t BrowOffSet = (side == 'R') ? 1 : -1;
  switch (position) {
    case (U): BrowOffSet *= 0; break;
    case (M): BrowOffSet *= BROW_OFFSET_MIDDLE; break;
    case (D): BrowOffSet *= BROW_OFFSET_DOWN; break;
  }

  if (side == 'R') {
    RBrowDest = 90 + BrowOffSet;
  } else {
    LBrowDest = 90 + BrowOffSet;
  }
}

// ======================================== updateServos ========================================//
// void updateServos() {
//   RShoulder.write(RShoulderDest);
//   LShoulder.write(LShoulderDest);
//   RHand.write(RHandDest);
//   LHand.write(LHandDest);
//   RNeck.write(RNeckDest);
//   LNeck.write(LNeckDest);
//   Ears.write(EarsDest);
//   RELid.write(RELidDest);
//   LELid.write(LELidDest);
//   RBrow.write(RBrowDest);
//   LBrow.write(LBrowDest);
// }

void updateServos() {
  RShoulder.write(RShoulderDest);
  delay(10); // Small delay between servo commands
  LShoulder.write(LShoulderDest);
  delay(10);
  RHand.write(RHandDest);
  delay(10);
  LHand.write(LHandDest);
  delay(10);
  RNeck.write(RNeckDest);
  delay(10);
  LNeck.write(LNeckDest);
  delay(10);
  Ears.write(EarsDest);
  delay(10);
  RELid.write(RELidDest);
  delay(10);
  LELid.write(LELidDest);
  delay(10);
  RBrow.write(RBrowDest);
  delay(10);
  LBrow.write(LBrowDest);
}

// ======================================== executeExpression =================================
bool executeExpression(String expression) {
  int moveSetIndex = getIndex(expression);
  if (moveSetIndex == -1) return false;

  if (millis() - timer > 900 || count == -1) {
    count++;
    timer = millis();
    if (CommandToInstruction(moveSetIndex, count)) {
      if (count > 3) count = -1;
    } else {
      count = -1;
    }
  }
  updateServos();

  return (count >= 0);
}  

// ======================================== CommandToInstruction ========================================//
bool CommandToInstruction(int moveSetIndex, int count) {
  setEyes('R', listOfMoveSets[moveSetIndex].REyeSet[count]);
  setEyes('L', listOfMoveSets[moveSetIndex].LEyeSet[count]);
  setBrows('R', listOfMoveSets[moveSetIndex].RBrowSet);
  setBrows('L', listOfMoveSets[moveSetIndex].LBrowSet);

  setEars(listOfMoveSets[moveSetIndex].EarsSet[count]);
  setNeck(listOfMoveSets[moveSetIndex].NeckSet[count]);

  setShoulder('R', listOfMoveSets[moveSetIndex].RShoulderSet[count]);
  setShoulder('L', listOfMoveSets[moveSetIndex].LShoulderSet[count]);

  setHand('R', listOfMoveSets[moveSetIndex].RHandSet[count]);
  setHand('L', listOfMoveSets[moveSetIndex].LHandSet[count]);

  // Check if there's a next step
  if (count + 1 >= SIZE_OF_SET) return false;
  if (listOfMoveSets[moveSetIndex].NeckSet[count + 1] == C) return false;

  return true;
}

// ======================================== getIndex ======================================== //
int getIndex(String input) {
  input.trim();
  input.toLowerCase();
  for (int i = 0; i < NUM_VALID_EXPRESSIONS; i++) {
    if (input.equals(validExpressions[i])) {
      return i;
    }
  }
  return -1;
}

// ======================================== checkValidity ========================================//
bool checkValidity(String input) {
  input.trim();
  input.toLowerCase();

  if (input.length() == 0) return false;

  for (int i = 0; i < NUM_VALID_EXPRESSIONS; i++) {
    if (input == validExpressions[i]) {
      return true;
    }
  }
  return false;
}