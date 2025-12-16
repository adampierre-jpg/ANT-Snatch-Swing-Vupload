/**

- KB Velocity Tracker (Snatch) — Fixed Version
- 
- FIXES:
- 1. Added set count tracking
- 1. Fixed rep detection with better phase state machine
- 1. Improved hand detection with clearer logic and hysteresis
- 1. Fixed velocity calculation (upward = positive velocity)
- 1. Added per-set peak tracking for velocity drop-off
- 1. Better lockout detection with arm extension check
- 1. Removed MIN_REPS_BEFORE_UNLOCK blocking issue
- 
- Landmark indices (MediaPipe Pose):
- wrists 15/16, elbows 13/14, shoulders 11/12, hips 23/24, knees 25/26, ankles 27/28
  */

import { PoseLandmarker, FilesetResolver } from “https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs”;

const CONFIG = {
LEFT:  { WRIST: 15, ELBOW: 13, SHOULDER: 11, HIP: 23, KNEE: 25, ANKLE: 27 },
RIGHT: { WRIST: 16, ELBOW: 14, SHOULDER: 12, HIP: 24, KNEE: 26, ANKLE: 28 },

MIN_DET_CONF: 0.5,
MIN_TRACK_CONF: 0.5,

// Physics
SMOOTHING_ALPHA: 0.25,
MAX_REALISTIC_VELOCITY: 12.0,
MIN_DT: 0.008,
MAX_DT: 0.12,
TORSO_METERS: 0.5,

// Rep detection thresholds
LOCKOUT_VY_THRESHOLD: 0.5,      // Max vertical velocity at lockout (m/s)
LOCKOUT_SPEED_THRESHOLD: 1.8,   // Max total speed at lockout (m/s)
OVERHEAD_HOLD_FRAMES: 2,        // Frames to confirm lockout

// Position thresholds (normalized, relative to body)
BOTTOM_THRESHOLD: 0.0,          // Wrist must be at or below hip level
OVERHEAD_THRESHOLD: 0.0,        // Wrist must be at or above shoulder level
ARM_EXTENSION_MIN: 0.85,        // Minimum arm extension ratio for lockout

// Drop-off thresholds
BASELINE_REPS: 3,
DROP_WARN: 15,
DROP_FAIL: 20,

// Floor-zone detection for hand grab
SHANK_FRACTION: 0.40,
MIN_SHANK_LEN_NORM: 0.05,
ARM_MS_REQUIRED: 150,           // Time wrist must be in floor zone to lock side

// Set detection (bell set down)
RESET_MS_REQUIRED: 200,
RESET_SPEED_CUTOFF: 0.6,
RESET_GRACE_MS: 800,            // Grace period after locking before allowing reset

// Debug
DEBUG_MODE: true
};

let state = {
video: null,
canvas: null,
ctx: null,
landmarker: null,

isModelLoaded: false,
isVideoReady: false,

lastPose: null,

// Test lifecycle
isTestRunning: false,
testStage: “IDLE”, // IDLE | ARMING | RUNNING

// Handedness detection
lockedSide: null,           // “left” | “right” | null
sideLocked: false,
lockedAtMs: 0,

// Floor dwell timers (for hand detection)
floorDwell: {
left: { inZone: false, startMs: null, dwellMs: 0 },
right: { inZone: false, startMs: null, dwellMs: 0 },
locked: { inZone: false, startMs: null, dwellMs: 0 }
},
lastUpdateMs: null,

// Physics tracking
prevWrist: null,
calibrationPxPerM: null,

// Smoothed values
smoothedSpeed: 0,
smoothedVy: 0,
rawSpeed: 0,
rawVy: 0,

// Rep state machine
phase: “IDLE”,              // IDLE | BOTTOM | ASCENDING | LOCKOUT
overheadHoldCount: 0,
currentRepPeak: 0,
currentRepStartMs: null,

// Totals (persist across sets)
totalReps: 0,
setCount: 0,
repHistory: [],             // All rep peaks
baseline: 0,

// Per-set tracking
currentSetReps: 0,
currentSetPeaks: [],
setHistory: []              // Array of {reps, avgPeak, startMs, endMs}
};

/* —————————– INIT —————————– */

async function initializeApp() {
try {
state.video = document.getElementById(“video”);
state.canvas = document.getElementById(“canvas”);
state.ctx = state.canvas.getContext(“2d”);

```
const visionGen = await FilesetResolver.forVisionTasks(
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
);

state.landmarker = await PoseLandmarker.createFromOptions(visionGen, {
  baseOptions: {
    modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    delegate: "GPU"
  },
  runningMode: "VIDEO",
  numPoses: 1,
  minPoseDetectionConfidence: CONFIG.MIN_DET_CONF,
  minTrackingConfidence: CONFIG.MIN_TRACK_CONF
});

state.isModelLoaded = true;
document.getElementById("loading-overlay")?.classList.add("hidden");
setStatus("Ready", "#3b82f6");

document.getElementById("btn-camera").onclick = startCamera;
document.getElementById("file-input").onchange = handleUpload;

state.video.addEventListener("loadeddata", onVideoReady);
state.video.addEventListener("error", () => {
  const err = state.video?.error;
  alert("Video error: " + (err?.message || "Unknown error"));
});

setupControls();
requestAnimationFrame(masterLoop);
```

} catch (e) {
console.error(e);
alert(“Startup Error: “ + e.message);
}
}

/* ––––––––––––– SOURCE HANDLING ––––––––––––– */

function handleUpload(e) {
const file = e.target.files?.[0];
if (!file) return;

resetAll(“Loading…”);
state.video.srcObject = null;
state.video.src = URL.createObjectURL(file);
state.video.load();
}

async function startCamera() {
try {
resetAll(“Starting camera…”);
const stream = await navigator.mediaDevices.getUserMedia({
video: { facingMode: “user” },
audio: false
});
state.video.srcObject = stream;
state.video.src = “”;
} catch (e) {
alert(“Camera Error: “ + e.message);
}
}

function onVideoReady() {
state.isVideoReady = true;
state.canvas.width = state.video.videoWidth || state.canvas.width;
state.canvas.height = state.video.videoHeight || state.canvas.height;
document.getElementById(“btn-start-test”).disabled = false;
primeVideo();
setStatus(“Video Loaded — press Start Test”, “#fbbf24”);
}

function primeVideo() {
const p = state.video.play();
if (p && typeof p.then === “function”) {
p.then(() => {
setTimeout(() => {
if (!state.isTestRunning) state.video.pause();
state.video.currentTime = 0;
}, 120);
}).catch(() => console.log(“Autoplay blocked.”));
}
}

/* –––––––––––––– MASTER LOOP –––––––––––––– */

function masterLoop() {
if (state.isModelLoaded && state.isVideoReady && state.video.readyState >= 2) {
const result = state.landmarker.detectForVideo(state.video, performance.now());

```
if (result.landmarks && result.landmarks.length > 0) {
  state.lastPose = result.landmarks[0];
  const tMs = state.video.currentTime * 1000;

  if (state.isTestRunning) {
    if (state.testStage === "ARMING") {
      detectHandGrab(state.lastPose, tMs);
    } else if (state.testStage === "RUNNING" && !state.video.paused) {
      processSnatchFrame(state.lastPose, tMs);
      detectSetDown(state.lastPose, tMs);
    }
  }
}
```

}

drawOverlay();
requestAnimationFrame(masterLoop);
}

/* —————– HAND DETECTION (ARMING PHASE) —————– */

function detectHandGrab(pose, tMs) {
const dtMs = getDeltaTime(tMs);
if (dtMs <= 0) return;

const leftInZone = isWristInFloorZone(pose, “left”);
const rightInZone = isWristInFloorZone(pose, “right”);

// Update dwell timers
updateDwellTimer(state.floorDwell.left, leftInZone, tMs, dtMs);
updateDwellTimer(state.floorDwell.right, rightInZone, tMs, dtMs);

const leftReady = state.floorDwell.left.dwellMs >= CONFIG.ARM_MS_REQUIRED;
const rightReady = state.floorDwell.right.dwellMs >= CONFIG.ARM_MS_REQUIRED;

// Lock side based on which hand is ready first, or deepest if both
if (leftReady && !rightReady) {
lockSide(“left”, tMs);
} else if (rightReady && !leftReady) {
lockSide(“right”, tMs);
} else if (leftReady && rightReady) {
// Both ready - pick the one that’s deeper/lower
const leftDepth = getFloorDepth(pose, “left”);
const rightDepth = getFloorDepth(pose, “right”);
lockSide(leftDepth >= rightDepth ? “left” : “right”, tMs);
}
}

function lockSide(side, tMs) {
state.lockedSide = side;
state.sideLocked = true;
state.lockedAtMs = tMs;

// Start new set
state.setCount++;
state.currentSetReps = 0;
state.currentSetPeaks = [];

// Reset mechanics for fresh tracking
resetMechanics();

// Reset floor timers
resetFloorDwellTimers();

state.testStage = “RUNNING”;
setStatus(`Set ${state.setCount}: ${side.toUpperCase()} hand — tracking`, “#10b981”);

updateSetDisplay();

if (CONFIG.DEBUG_MODE) {
console.log(`[SET ${state.setCount}] Locked ${side} at ${tMs.toFixed(0)}ms`);
}
}

/* —————– SET DOWN DETECTION —————– */

function detectSetDown(pose, tMs) {
if (!state.sideLocked || !state.lockedSide) return;

// Grace period after locking
if ((tMs - state.lockedAtMs) < CONFIG.RESET_GRACE_MS) return;

const dtMs = getDeltaTime(tMs);
if (dtMs <= 0) return;

const inFloorZone = isWristInFloorZone(pose, state.lockedSide);
const isSlowEnough = state.smoothedSpeed < CONFIG.RESET_SPEED_CUTOFF;
const shouldCount = inFloorZone && isSlowEnough;

updateDwellTimer(state.floorDwell.locked, shouldCount, tMs, dtMs);

if (state.floorDwell.locked.dwellMs >= CONFIG.RESET_MS_REQUIRED) {
// End current set
if (state.currentSetReps > 0) {
state.setHistory.push({
reps: state.currentSetReps,
peaks: […state.currentSetPeaks],
avgPeak: avg(state.currentSetPeaks),
side: state.lockedSide,
startMs: state.lockedAtMs,
endMs: tMs
});
}

```
if (CONFIG.DEBUG_MODE) {
  console.log(`[SET ${state.setCount} END] ${state.currentSetReps} reps, avg peak: ${avg(state.currentSetPeaks).toFixed(2)} m/s`);
}

// Unlock for next set
state.sideLocked = false;
state.lockedSide = null;
state.lockedAtMs = 0;

// Reset for re-arming
resetFloorDwellTimers();
resetMechanics();

state.testStage = "ARMING";
setStatus("Set complete — grab bell to start next set", "#fbbf24");
```

}
}

/* —————– SNATCH PHYSICS + REP DETECTION —————– */

function processSnatchFrame(pose, tMs) {
if (!state.lockedSide) return;

const idx = state.lockedSide === “left” ? CONFIG.LEFT : CONFIG.RIGHT;
const wrist = pose[idx.WRIST];
const elbow = pose[idx.ELBOW];
const shoulder = pose[idx.SHOULDER];
const hip = pose[idx.HIP];

if (!wrist || !shoulder || !hip) return;
if ((wrist.visibility ?? 1) < 0.4) return;

// Calibration (pixels per meter based on torso length)
if (!state.calibrationPxPerM) {
const torsoPx = Math.abs(shoulder.y - hip.y) * state.canvas.height;
state.calibrationPxPerM = torsoPx > 10 ? torsoPx / CONFIG.TORSO_METERS : 200;
}

// Calculate velocity
const velocity = calculateVelocity(wrist, tMs);
if (!velocity) return;

const { speed, vy } = velocity;

// Update smoothed values
state.rawSpeed = speed;
state.rawVy = vy;
state.smoothedSpeed = CONFIG.SMOOTHING_ALPHA * speed + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedSpeed;
state.smoothedVy = CONFIG.SMOOTHING_ALPHA * vy + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVy;

// Update velocity display
document.getElementById(“val-velocity”).textContent = state.smoothedSpeed.toFixed(2);

// Determine position relative to body
const isBelowHip = wrist.y >= hip.y + CONFIG.BOTTOM_THRESHOLD;
const isAboveShoulder = wrist.y <= shoulder.y + CONFIG.OVERHEAD_THRESHOLD;

// Check arm extension (for lockout validation)
const armExtension = elbow ? getArmExtension(wrist, elbow, shoulder) : 1.0;
const isArmExtended = armExtension >= CONFIG.ARM_EXTENSION_MIN;

// Rep detection state machine
detectRep(isBelowHip, isAboveShoulder, isArmExtended, tMs);

// Store wrist for next frame
state.prevWrist = { x: wrist.x, y: wrist.y, tMs };
}

function calculateVelocity(wrist, tMs) {
if (!state.prevWrist) {
state.prevWrist = { x: wrist.x, y: wrist.y, tMs };
return null;
}

const dt = (tMs - state.prevWrist.tMs) / 1000;
if (dt < CONFIG.MIN_DT || dt > CONFIG.MAX_DT) {
state.prevWrist = { x: wrist.x, y: wrist.y, tMs };
return null;
}

// Convert normalized coordinates to pixels, then to meters
const dxPx = (wrist.x - state.prevWrist.x) * state.canvas.width;
const dyPx = (wrist.y - state.prevWrist.y) * state.canvas.height;

const dxM = dxPx / state.calibrationPxPerM;
const dyM = dyPx / state.calibrationPxPerM;

const vx = dxM / dt;
// Note: In MediaPipe, Y increases downward. Invert so upward motion = positive vy
const vy = -dyM / dt;
const speed = Math.hypot(vx, vy);

// Reject unrealistic velocities
if (speed > CONFIG.MAX_REALISTIC_VELOCITY) {
return null;
}

return { speed, vx, vy };
}

function getArmExtension(wrist, elbow, shoulder) {
// Calculate how extended the arm is (1.0 = fully extended)
const upperArmLen = Math.hypot(elbow.x - shoulder.x, elbow.y - shoulder.y);
const forearmLen = Math.hypot(wrist.x - elbow.x, wrist.y - elbow.y);
const totalArmLen = Math.hypot(wrist.x - shoulder.x, wrist.y - shoulder.y);

const maxLen = upperArmLen + forearmLen;
return maxLen > 0 ? totalArmLen / maxLen : 0;
}

function detectRep(isBelowHip, isAboveShoulder, isArmExtended, tMs) {
const speed = state.smoothedSpeed;
const vy = state.smoothedVy;

switch (state.phase) {
case “IDLE”:
// Wait for wrist to go to bottom position to start rep cycle
if (isBelowHip) {
state.phase = “BOTTOM”;
state.currentRepPeak = 0;
state.currentRepStartMs = tMs;
if (CONFIG.DEBUG_MODE) console.log(`[PHASE] IDLE → BOTTOM`);
}
break;

```
case "BOTTOM":
  // Track peak velocity while at bottom
  if (speed > state.currentRepPeak) {
    state.currentRepPeak = speed;
  }
  
  // Transition to ascending when wrist leaves bottom zone AND moving upward
  if (!isBelowHip && vy > 0.5) {
    state.phase = "ASCENDING";
    if (CONFIG.DEBUG_MODE) console.log(`[PHASE] BOTTOM → ASCENDING (vy: ${vy.toFixed(2)})`);
  }
  break;

case "ASCENDING":
  // Continue tracking peak velocity during ascent
  if (speed > state.currentRepPeak) {
    state.currentRepPeak = speed;
  }

  // Check for lockout conditions
  const lockoutConditions = {
    overhead: isAboveShoulder,
    slowVy: Math.abs(vy) < CONFIG.LOCKOUT_VY_THRESHOLD,
    slowSpeed: speed < CONFIG.LOCKOUT_SPEED_THRESHOLD,
    armExtended: isArmExtended
  };

  const isLockout = lockoutConditions.overhead && 
                    lockoutConditions.slowVy && 
                    lockoutConditions.slowSpeed;

  if (isLockout) {
    state.overheadHoldCount++;
    
    if (state.overheadHoldCount >= CONFIG.OVERHEAD_HOLD_FRAMES) {
      completeRep(tMs);
    }
  } else {
    state.overheadHoldCount = 0;
    
    // If we drop back below hip without completing, it's a failed rep
    if (isBelowHip) {
      state.phase = "BOTTOM";
      state.currentRepPeak = 0;
      if (CONFIG.DEBUG_MODE) console.log(`[PHASE] ASCENDING → BOTTOM (failed rep)`);
    }
  }
  break;

case "LOCKOUT":
  // Wait for wrist to drop back down to start next rep
  if (isBelowHip) {
    state.phase = "BOTTOM";
    state.currentRepPeak = 0;
    state.currentRepStartMs = tMs;
    state.overheadHoldCount = 0;
    if (CONFIG.DEBUG_MODE) console.log(`[PHASE] LOCKOUT → BOTTOM`);
  }
  break;
```

}
}

function completeRep(tMs) {
state.phase = “LOCKOUT”;
state.overheadHoldCount = 0;

// Update counts
state.totalReps++;
state.currentSetReps++;

// Store peak velocity
state.repHistory.push(state.currentRepPeak);
state.currentSetPeaks.push(state.currentRepPeak);

if (CONFIG.DEBUG_MODE) {
console.log(`[REP ${state.totalReps}] Peak: ${state.currentRepPeak.toFixed(2)} m/s (Set ${state.setCount}, Rep ${state.currentSetReps})`);
}

// Update displays
document.getElementById(“val-reps”).textContent = String(state.totalReps);
document.getElementById(“val-peak”).textContent = state.currentRepPeak.toFixed(2);
updateSetDisplay();

// Calculate and display drop-off
updateDropOff();
}

function updateDropOff() {
const dropEl = document.getElementById(“val-drop”);

if (state.repHistory.length <= CONFIG.BASELINE_REPS) {
// Still building baseline
state.baseline = avg(state.repHistory);
dropEl.textContent = “CALC…”;
dropEl.style.color = “#94a3b8”;
} else {
// Compare current rep to baseline
const currentPeak = state.repHistory[state.repHistory.length - 1];
const dropFraction = (state.baseline - currentPeak) / state.baseline;
const dropPct = dropFraction * 100;

```
dropEl.textContent = dropPct > 0 ? `-${dropPct.toFixed(1)}%` : `+${Math.abs(dropPct).toFixed(1)}%`;

if (dropPct >= CONFIG.DROP_FAIL) {
  dropEl.style.color = "#ef4444"; // Red
} else if (dropPct >= CONFIG.DROP_WARN) {
  dropEl.style.color = "#fbbf24"; // Yellow
} else {
  dropEl.style.color = "#10b981"; // Green
}
```

}
}

/* —————– FLOOR ZONE HELPERS —————– */

function isWristInFloorZone(pose, side) {
const idx = side === “left” ? CONFIG.LEFT : CONFIG.RIGHT;
const wrist = pose[idx.WRIST];
const knee = pose[idx.KNEE];
const ankle = pose[idx.ANKLE];

if (!wrist || !knee || !ankle) return false;
if ((wrist.visibility ?? 1) < 0.4) return false;

// Shank length (knee to ankle)
const shankLen = ankle.y - knee.y;
if (shankLen < CONFIG.MIN_SHANK_LEN_NORM) return false;

// Floor zone threshold: below knee by SHANK_FRACTION of shank length
const threshold = knee.y + CONFIG.SHANK_FRACTION * shankLen;
return wrist.y > threshold;
}

function getFloorDepth(pose, side) {
const idx = side === “left” ? CONFIG.LEFT : CONFIG.RIGHT;
const wrist = pose[idx.WRIST];
const knee = pose[idx.KNEE];
const ankle = pose[idx.ANKLE];

if (!wrist || !knee || !ankle) return 0;

const shankLen = ankle.y - knee.y;
if (shankLen < CONFIG.MIN_SHANK_LEN_NORM) return 0;

const threshold = knee.y + CONFIG.SHANK_FRACTION * shankLen;
return Math.max(0, wrist.y - threshold);
}

function updateDwellTimer(timer, isInZone, tMs, dtMs) {
if (isInZone) {
if (!timer.inZone) {
timer.inZone = true;
timer.startMs = tMs;
timer.dwellMs = 0;
}
timer.dwellMs += dtMs;
} else {
timer.inZone = false;
timer.startMs = null;
timer.dwellMs = 0;
}
}

function resetFloorDwellTimers() {
state.floorDwell.left = { inZone: false, startMs: null, dwellMs: 0 };
state.floorDwell.right = { inZone: false, startMs: null, dwellMs: 0 };
state.floorDwell.locked = { inZone: false, startMs: null, dwellMs: 0 };
}

function getDeltaTime(tMs) {
if (state.lastUpdateMs == null) {
state.lastUpdateMs = tMs;
return 0;
}
const dtMs = tMs - state.lastUpdateMs;
state.lastUpdateMs = tMs;

// Reject bad deltas (negative or too large)
if (dtMs < 0 || dtMs > 250) return 0;
return dtMs;
}

/* —————————— DRAWING —————————— */

function drawOverlay() {
state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
if (!state.lastPose) return;

if (state.isTestRunning && state.testStage === “ARMING”) {
drawArmingOverlay(state.lastPose);
} else if (state.isTestRunning && state.testStage === “RUNNING”) {
drawTrackingOverlay(state.lastPose);
}
}

function drawArmingOverlay(pose) {
// Draw floor zone lines for both sides
drawFloorZoneLine(pose, “left”, “#60a5fa”);
drawFloorZoneLine(pose, “right”, “#f87171”);

// Draw wrist positions with dwell progress
const leftWrist = pose[CONFIG.LEFT.WRIST];
const rightWrist = pose[CONFIG.RIGHT.WRIST];

if (leftWrist) {
const progress = Math.min(1, state.floorDwell.left.dwellMs / CONFIG.ARM_MS_REQUIRED);
drawWristDot(leftWrist, “#60a5fa”, progress);
}
if (rightWrist) {
const progress = Math.min(1, state.floorDwell.right.dwellMs / CONFIG.ARM_MS_REQUIRED);
drawWristDot(rightWrist, “#f87171”, progress);
}
}

function drawTrackingOverlay(pose) {
if (!state.lockedSide) return;

const idx = state.lockedSide === “left” ? CONFIG.LEFT : CONFIG.RIGHT;
const wrist = pose[idx.WRIST];
const shoulder = pose[idx.SHOULDER];
const hip = pose[idx.HIP];

// Draw body reference lines
if (shoulder && hip) {
drawHorizontalLine(shoulder.y, “#10b981”, “Shoulder”);
drawHorizontalLine(hip.y, “#3b82f6”, “Hip”);
}

// Draw wrist position
if (wrist) {
const color = state.phase === “LOCKOUT” ? “#10b981” :
state.phase === “ASCENDING” ? “#fbbf24” : “#3b82f6”;
drawWristDot(wrist, color, 1.0, true);
}

// Draw phase indicator
drawPhaseIndicator();
}

function drawFloorZoneLine(pose, side, color) {
const idx = side === “left” ? CONFIG.LEFT : CONFIG.RIGHT;
const knee = pose[idx.KNEE];
const ankle = pose[idx.ANKLE];

if (!knee || !ankle) return;

const shankLen = ankle.y - knee.y;
if (shankLen < CONFIG.MIN_SHANK_LEN_NORM) return;

const threshold = knee.y + CONFIG.SHANK_FRACTION * shankLen;
const yPx = threshold * state.canvas.height;

state.ctx.strokeStyle = color;
state.ctx.globalAlpha = 0.5;
state.ctx.lineWidth = 2;
state.ctx.setLineDash([5, 5]);
state.ctx.beginPath();
state.ctx.moveTo(0, yPx);
state.ctx.lineTo(state.canvas.width, yPx);
state.ctx.stroke();
state.ctx.setLineDash([]);
state.ctx.globalAlpha = 1.0;
}

function drawHorizontalLine(yNorm, color, label) {
const yPx = yNorm * state.canvas.height;

state.ctx.strokeStyle = color;
state.ctx.globalAlpha = 0.6;
state.ctx.lineWidth = 2;
state.ctx.beginPath();
state.ctx.moveTo(0, yPx);
state.ctx.lineTo(state.canvas.width, yPx);
state.ctx.stroke();
state.ctx.globalAlpha = 1.0;

if (label) {
state.ctx.fillStyle = color;
state.ctx.font = “12px sans-serif”;
state.ctx.fillText(label, 5, yPx - 5);
}
}

function drawWristDot(wrist, color, progress = 1.0, large = false) {
const x = wrist.x * state.canvas.width;
const y = wrist.y * state.canvas.height;
const radius = large ? 12 : 8;

// Progress ring
if (progress < 1 && progress > 0) {
state.ctx.strokeStyle = color;
state.ctx.lineWidth = 3;
state.ctx.beginPath();
state.ctx.arc(x, y, radius + 4, -Math.PI/2, -Math.PI/2 + (2 * Math.PI * progress));
state.ctx.stroke();
}

// Main dot
state.ctx.fillStyle = color;
state.ctx.beginPath();
state.ctx.arc(x, y, radius, 0, 2 * Math.PI);
state.ctx.fill();

// Center dot
state.ctx.fillStyle = “white”;
state.ctx.beginPath();
state.ctx.arc(x, y, 3, 0, 2 * Math.PI);
state.ctx.fill();
}

function drawPhaseIndicator() {
const phases = {
“IDLE”: { color: “#94a3b8”, label: “IDLE” },
“BOTTOM”: { color: “#3b82f6”, label: “BOTTOM” },
“ASCENDING”: { color: “#fbbf24”, label: “ASCENDING” },
“LOCKOUT”: { color: “#10b981”, label: “LOCKOUT ✓” }
};

const p = phases[state.phase] || phases[“IDLE”];

state.ctx.fillStyle = p.color;
state.ctx.font = “bold 14px sans-serif”;
state.ctx.fillText(`Phase: ${p.label}`, 10, 25);

// Current set info
state.ctx.fillStyle = “#f1f5f9”;
state.ctx.font = “12px sans-serif”;
state.ctx.fillText(`Set ${state.setCount} | ${state.lockedSide?.toUpperCase() || '?'} | Rep ${state.currentSetReps}`, 10, 45);
}

/* —————————— CONTROLS —————————— */

function setupControls() {
const startBtn = document.getElementById(“btn-start-test”);
const resetBtn = document.getElementById(“btn-reset”);

startBtn.onclick = async () => {
if (!state.isVideoReady) return;

```
state.isTestRunning = true;
state.testStage = "ARMING";

// Reset side detection
state.sideLocked = false;
state.lockedSide = null;
state.lockedAtMs = 0;

// Reset floor timers
resetFloorDwellTimers();
state.lastUpdateMs = null;

// Reset mechanics
resetMechanics();

startBtn.textContent = "Test Running…";
startBtn.disabled = true;
resetBtn.disabled = false;

setStatus("ARMING: Grab bell to lock side and start tracking…", "#fbbf24");

try {
  await state.video.play();
} catch {
  alert("Playback blocked. Tap the video area once, then press Start Test again.");
  startBtn.textContent = "▶ Start Test";
  startBtn.disabled = false;
  resetBtn.disabled = true;
  state.isTestRunning = false;
  state.testStage = "IDLE";
}
```

};

resetBtn.onclick = () => {
state.video.pause();
state.video.currentTime = 0;
resetAll(“Reset — ready”);
primeVideo();
};
}

/* —————————— RESET HELPERS —————————— */

function resetMechanics() {
state.prevWrist = null;
state.calibrationPxPerM = null;
state.smoothedSpeed = 0;
state.smoothedVy = 0;
state.rawSpeed = 0;
state.rawVy = 0;
state.phase = “IDLE”;
state.overheadHoldCount = 0;
state.currentRepPeak = 0;
state.currentRepStartMs = null;

document.getElementById(“val-velocity”).textContent = “0.00”;
}

function resetAll(statusText) {
state.isTestRunning = false;
state.testStage = “IDLE”;

state.sideLocked = false;
state.lockedSide = null;
state.lockedAtMs = 0;

resetFloorDwellTimers();
state.lastUpdateMs = null;

// Clear all totals
state.totalReps = 0;
state.setCount = 0;
state.repHistory = [];
state.baseline = 0;
state.currentSetReps = 0;
state.currentSetPeaks = [];
state.setHistory = [];

resetMechanics();

// Reset displays
document.getElementById(“val-reps”).textContent = “0”;
document.getElementById(“val-peak”).textContent = “0.00”;
updateSetDisplay();

const dropEl = document.getElementById(“val-drop”);
dropEl.textContent = “–”;
dropEl.style.color = “#f1f5f9”;

const startBtn = document.getElementById(“btn-start-test”);
const resetBtn = document.getElementById(“btn-reset”);
startBtn.textContent = “▶ Start Test”;
startBtn.disabled = false;
resetBtn.disabled = true;

setStatus(statusText, “#3b82f6”);
}

function updateSetDisplay() {
const setEl = document.getElementById(“val-sets”);
if (setEl) {
setEl.textContent = String(state.setCount);
}
}

/* —————————— UTILS —————————— */

function avg(arr) {
if (!arr || arr.length === 0) return 0;
return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function setStatus(text, color) {
const pill = document.getElementById(“status-pill”);
if (pill) {
pill.textContent = text;
pill.style.color = color;
pill.style.borderColor = color;
}
}

initializeApp();