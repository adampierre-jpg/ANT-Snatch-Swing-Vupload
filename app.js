/**

- KB Velocity Tracker (Snatch) — Fixed Version
- 
- FIXES from original:
- 1. Added set count tracking
- 1. Relaxed lockout thresholds (were too strict)
- 1. Fixed velocity sign (upward = positive)
- 1. Improved phase detection with better transitions
- 1. Added debug output to diagnose issues
- 
- Landmark indices (MediaPipe Pose): wrists 15/16, hips 23/24, knees 25/26, ankles 27/28, shoulders 11/12
  */

import { PoseLandmarker, FilesetResolver } from “https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs”;

const CONFIG = {
LEFT:  { WRIST: 15, SHOULDER: 11, HIP: 23, KNEE: 25, ANKLE: 27 },
RIGHT: { WRIST: 16, SHOULDER: 12, HIP: 24, KNEE: 26, ANKLE: 28 },

MIN_DET_CONF: 0.5,
MIN_TRACK_CONF: 0.5,

// Physics
SMOOTHING_ALPHA: 0.30,
MAX_REALISTIC_VELOCITY: 12.0,
MIN_DT: 0.008,
MAX_DT: 0.12,
TORSO_METERS: 0.5,

// Rep logic - RELAXED thresholds
LOCKOUT_VY_CUTOFF: 1.2,       // Was 0.40 - too strict
LOCKOUT_SPEED_CUTOFF: 2.5,    // Was 1.40 - too strict
OVERHEAD_HOLD_FRAMES: 2,

// Drop-off
BASELINE_REPS: 3,
DROP_WARN: 15,
DROP_FAIL: 20,

// Floor-zone “hand on bell”
SHANK_FRACTION: 0.35,
MIN_SHANK_LEN_NORM: 0.05,
ARM_MS_REQUIRED: 100,

// Reset (“bell set down”)
RESET_MS_REQUIRED: 150,
RESET_SPEED_CUTOFF: 0.80,
RESET_VY_CUTOFF: 0.80,
MIN_REPS_BEFORE_UNLOCK: 1,    // Was 2 - allow quicker hand switch
RESET_GRACE_MS_AFTER_LOCK: 400
};

let state = {
video: null,
canvas: null,
ctx: null,
landmarker: null,

isModelLoaded: false,
isVideoReady: false,

lastPose: null,

// test lifecycle
isTestRunning: false,
testStage: “IDLE”, // IDLE | ARMING | RUNNING

// handedness
lockedSide: “unknown”,
sideLocked: false,
lockedAtMs: 0,

// floor dwell timers
floorMsLeft: 0,
floorMsRight: 0,
floorMsLocked: 0,
lastFloorUpdateMs: null,

// physics (per-set mechanics)
prevWrist: null,
lockedCalibration: null,
smoothedVelocity: null,
smoothedVy: null,
lastSpeed: 0,
lastVy: 0,

// rep totals (persist for whole video)
phase: “IDLE”,
repCount: 0,
currentRepPeak: 0,
repHistory: [],
baseline: 0,

overheadHoldCount: 0,

// NEW: Set tracking
setCount: 0,
currentSetReps: 0
};

/* —————————– DEBUG —————————– */

function dbg(id, val) {
const el = document.getElementById(id);
if (el) el.textContent = val;
}

/* —————————– INIT —————————– */

async function initializeApp() {
console.log(”[INIT] Starting…”);

try {
state.video = document.getElementById(“video”);
state.canvas = document.getElementById(“canvas”);
state.ctx = state.canvas.getContext(“2d”);

```
console.log("[INIT] Loading FilesetResolver...");
const visionGen = await FilesetResolver.forVisionTasks(
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
);
console.log("[INIT] FilesetResolver loaded");

console.log("[INIT] Creating PoseLandmarker...");
state.landmarker = await PoseLandmarker.createFromOptions(visionGen, {
  baseOptions: {
    modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    delegate: "GPU"
  },
  runningMode: "VIDEO",
  numPoses: 1,
  minPoseDetectionConfidence: CONFIG.MIN_DET_CONF,
  minTrackingConfidence: CONFIG.MIN_TRACK_CONF
});
console.log("[INIT] PoseLandmarker created");

state.isModelLoaded = true;
document.getElementById("loading-overlay").classList.add("hidden");
setStatus("Ready", "#3b82f6");

document.getElementById("btn-camera").onclick = startCamera;
document.getElementById("file-input").onchange = handleUpload;

state.video.addEventListener("loadeddata", onVideoReady);
state.video.addEventListener("error", () => {
  const err = state.video?.error;
  console.error("[VIDEO ERROR]", err);
  alert("Video error: " + (err?.message || "Unknown error"));
});

setupControls();
requestAnimationFrame(masterLoop);
console.log("[INIT] Complete");
```

} catch (e) {
console.error(”[INIT ERROR]”, e);
alert(“Startup Error: “ + e.message);
}
}

/* ––––––––––––– SOURCE HANDLING ––––––––––––– */

function handleUpload(e) {
const file = e.target.files?.[0];
if (!file) {
console.log(”[UPLOAD] No file selected”);
return;
}

console.log(”[UPLOAD] File:”, file.name, file.type, file.size);
resetAll(“Loading…”);

// Clear any existing source
state.video.srcObject = null;

// Create object URL and set as source
const url = URL.createObjectURL(file);
console.log(”[UPLOAD] Object URL:”, url);

state.video.src = url;
state.video.load();
console.log(”[UPLOAD] Called video.load()”);
}

async function startCamera() {
try {
console.log(”[CAMERA] Requesting…”);
resetAll(“Starting camera…”);
const stream = await navigator.mediaDevices.getUserMedia({
video: { facingMode: “user” },
audio: false
});
console.log(”[CAMERA] Got stream”);
state.video.srcObject = stream;
state.video.src = “”;
state.video.play();
} catch (e) {
console.error(”[CAMERA ERROR]”, e);
alert(“Camera Error: “ + e.message);
}
}

function onVideoReady() {
console.log(”[VIDEO] loadeddata event fired”);
console.log(”[VIDEO] dimensions:”, state.video.videoWidth, “x”, state.video.videoHeight);
console.log(”[VIDEO] duration:”, state.video.duration);

state.isVideoReady = true;

state.canvas.width = state.video.videoWidth || 640;
state.canvas.height = state.video.videoHeight || 480;

document.getElementById(“btn-start-test”).disabled = false;

primeVideo();
setStatus(“Video Loaded — press Start Test”, “#fbbf24”);
}

function primeVideo() {
console.log(”[VIDEO] Priming…”);
const p = state.video.play();
if (p && typeof p.then === “function”) {
p.then(() => {
console.log(”[VIDEO] Play started for priming”);
setTimeout(() => {
if (!state.isTestRunning) {
state.video.pause();
console.log(”[VIDEO] Paused after prime”);
}
state.video.currentTime = 0;
}, 150);
}).catch((err) => {
console.log(”[VIDEO] Autoplay blocked:”, err.message);
});
}
}

/* –––––––––––––– MASTER LOOP –––––––––––––– */

function masterLoop() {
if (state.isModelLoaded && state.isVideoReady && state.video.readyState >= 2) {
try {
const result = state.landmarker.detectForVideo(state.video, performance.now());

```
  if (result.landmarks && result.landmarks.length > 0) {
    state.lastPose = result.landmarks[0];
    const tMs = state.video.currentTime * 1000;

    if (state.isTestRunning && state.testStage === "ARMING") {
      updateHandednessByFloorGrab(state.lastPose, tMs);
    }

    if (state.isTestRunning && state.testStage === "RUNNING" && !state.video.paused) {
      runSnatchPhysicsAndLogic(state.lastPose, tMs);
      maybeResetOnSetDown(state.lastPose, tMs);
    }
  }
} catch (e) {
  console.error("[LOOP ERROR]", e);
}
```

}

drawOverlay();
requestAnimationFrame(masterLoop);
}

/* —————– FLOOR ZONE: ARMING + RESET —————– */

function updateHandednessByFloorGrab(pose, tMs) {
const dtMs = updateFloorDt(tMs);
if (dtMs <= 0) return;

const inFloorLeft = isWristInFloorZone(pose, “left”);
const inFloorRight = isWristInFloorZone(pose, “right”);

state.floorMsLeft = inFloorLeft ? (state.floorMsLeft + dtMs) : 0;
state.floorMsRight = inFloorRight ? (state.floorMsRight + dtMs) : 0;

// Debug output
dbg(“dbg-side”, `L:${state.floorMsLeft.toFixed(0)} R:${state.floorMsRight.toFixed(0)}`);

if (state.floorMsLeft >= CONFIG.ARM_MS_REQUIRED && state.floorMsRight < CONFIG.ARM_MS_REQUIRED) {
lockSideAndStartRunning(“left”, tMs);
return;
}
if (state.floorMsRight >= CONFIG.ARM_MS_REQUIRED && state.floorMsLeft < CONFIG.ARM_MS_REQUIRED) {
lockSideAndStartRunning(“right”, tMs);
return;
}

if (state.floorMsLeft >= CONFIG.ARM_MS_REQUIRED && state.floorMsRight >= CONFIG.ARM_MS_REQUIRED) {
const depthL = floorDepth(pose, “left”);
const depthR = floorDepth(pose, “right”);
lockSideAndStartRunning(depthL >= depthR ? “left” : “right”, tMs);
}
}

function lockSideAndStartRunning(side, tMs) {
console.log(”[LOCK]”, side, “at”, tMs);

state.lockedSide = side;
state.sideLocked = true;
state.lockedAtMs = tMs;

// NEW: Increment set count
state.setCount++;
state.currentSetReps = 0;
document.getElementById(“val-sets”).textContent = String(state.setCount);

resetSetMechanicsOnly();
state.floorMsLocked = 0;

state.testStage = “RUNNING”;
setStatus(`Set ${state.setCount}: ${side.toUpperCase()} — tracking`, “#10b981”);

dbg(“dbg-stage”, “RUNNING”);
dbg(“dbg-side”, side.toUpperCase());
}

function maybeResetOnSetDown(pose, tMs) {
if (!state.sideLocked) return;

if (state.currentSetReps < CONFIG.MIN_REPS_BEFORE_UNLOCK) return;

if ((tMs - state.lockedAtMs) < CONFIG.RESET_GRACE_MS_AFTER_LOCK) return;

const dtMs = updateFloorDt(tMs);
if (dtMs <= 0) return;

const inFloorLocked = isWristInFloorZone(pose, state.lockedSide);

const slowEnough =
(state.lastSpeed < CONFIG.RESET_SPEED_CUTOFF) &&
(Math.abs(state.lastVy) < CONFIG.RESET_VY_CUTOFF);

const countThisFrame = inFloorLocked && slowEnough;
state.floorMsLocked = countThisFrame ? (state.floorMsLocked + dtMs) : 0;

if (state.floorMsLocked >= CONFIG.RESET_MS_REQUIRED) {
console.log(”[SET DOWN] Detected after”, state.currentSetReps, “reps”);

```
state.sideLocked = false;
state.lockedSide = "unknown";
state.lockedAtMs = 0;

state.floorMsLeft = 0;
state.floorMsRight = 0;
state.floorMsLocked = 0;
state.lastFloorUpdateMs = tMs;

resetSetMechanicsOnly();

state.testStage = "ARMING";
setStatus("Set complete — grab bell to start next set", "#fbbf24");
dbg("dbg-stage", "ARMING");
```

}
}

function updateFloorDt(tMs) {
if (state.lastFloorUpdateMs == null) {
state.lastFloorUpdateMs = tMs;
return 0;
}
const dtMs = tMs - state.lastFloorUpdateMs;
state.lastFloorUpdateMs = tMs;
if (dtMs < 0 || dtMs > 250) return 0;
return dtMs;
}

function isWristInFloorZone(pose, side) {
const idx = side === “left” ? CONFIG.LEFT : CONFIG.RIGHT;
const wrist = pose[idx.WRIST];
const knee = pose[idx.KNEE];
const ankle = pose[idx.ANKLE];
if (!wrist || !knee || !ankle) return false;

const shank = ankle.y - knee.y;
if (shank < CONFIG.MIN_SHANK_LEN_NORM) return false;

const threshold = knee.y + CONFIG.SHANK_FRACTION * shank;
return wrist.y > threshold;
}

function floorDepth(pose, side) {
const idx = side === “left” ? CONFIG.LEFT : CONFIG.RIGHT;
const wrist = pose[idx.WRIST];
const knee = pose[idx.KNEE];
const ankle = pose[idx.ANKLE];
if (!wrist || !knee || !ankle) return 0;

const shank = ankle.y - knee.y;
if (shank < CONFIG.MIN_SHANK_LEN_NORM) return 0;

const threshold = knee.y + CONFIG.SHANK_FRACTION * shank;
return Math.max(0, wrist.y - threshold);
}

/* ––––––––––– SNATCH PHYSICS + LOGIC ––––––––––– */

function runSnatchPhysicsAndLogic(pose, timeMs) {
const side = state.lockedSide;
if (side !== “left” && side !== “right”) return;

const idx = side === “left” ? CONFIG.LEFT : CONFIG.RIGHT;
const wrist = pose[idx.WRIST];
const shoulder = pose[idx.SHOULDER];
const hip = pose[idx.HIP];

if (!wrist || !shoulder || !hip) return;
if ((wrist.visibility ?? 1) < 0.4) return;

// Calibration
if (!state.lockedCalibration) {
const torsoPx = Math.abs(shoulder.y - hip.y) * state.canvas.height;
state.lockedCalibration = torsoPx > 0 ? torsoPx / CONFIG.TORSO_METERS : 100;
}

// First frame - just store position
if (!state.prevWrist) {
state.prevWrist = { xNorm: wrist.x, yNorm: wrist.y, tMs: timeMs };
return;
}

const dt = (timeMs - state.prevWrist.tMs) / 1000;
if (dt < CONFIG.MIN_DT || dt > CONFIG.MAX_DT) {
state.prevWrist = { xNorm: wrist.x, yNorm: wrist.y, tMs: timeMs };
return;
}

// Calculate velocity
const dxPx = (wrist.x - state.prevWrist.xNorm) * state.canvas.width;
const dyPx = (wrist.y - state.prevWrist.yNorm) * state.canvas.height;

const vx = (dxPx / state.lockedCalibration) / dt;
// FIX: Invert vy so upward motion = positive
const vy = -(dyPx / state.lockedCalibration) / dt;
const speed = Math.hypot(vx, vy);

state.lastSpeed = speed;
state.lastVy = vy;

if (speed > CONFIG.MAX_REALISTIC_VELOCITY) {
state.prevWrist = { xNorm: wrist.x, yNorm: wrist.y, tMs: timeMs };
return;
}

// Smoothing
if (state.smoothedVelocity == null) state.smoothedVelocity = speed;
if (state.smoothedVy == null) state.smoothedVy = vy;

state.smoothedVelocity =
CONFIG.SMOOTHING_ALPHA * speed + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVelocity;
state.smoothedVy =
CONFIG.SMOOTHING_ALPHA * vy + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVy;

const v = state.smoothedVelocity;
const vyS = state.smoothedVy;

document.getElementById(“val-velocity”).textContent = v.toFixed(2);
dbg(“dbg-vy”, vyS.toFixed(2));
dbg(“dbg-spd”, v.toFixed(2));
dbg(“dbg-wrist”, wrist.y.toFixed(2));

// Position checks (remember: Y increases downward in MediaPipe)
const isBelowHip = wrist.y > hip.y;
const isAboveShoulder = wrist.y < shoulder.y;

// Debug phase
dbg(“dbg-phase”, state.phase);

// Phase state machine
if (state.phase === “IDLE” || state.phase === “LOCKOUT”) {
if (isBelowHip) {
state.phase = “BOTTOM”;
state.overheadHoldCount = 0;
console.log(”[PHASE] -> BOTTOM”);
}
} else if (state.phase === “BOTTOM”) {
// Track peak even at bottom (catch the pull)
if (v > state.currentRepPeak) state.currentRepPeak = v;

```
// Transition when wrist leaves bottom AND moving upward
if (!isBelowHip && vyS > 0.3) {
  state.phase = "CONCENTRIC";
  console.log("[PHASE] -> CONCENTRIC, peak so far:", state.currentRepPeak.toFixed(2));
}
```

} else if (state.phase === “CONCENTRIC”) {
// Continue tracking peak
if (v > state.currentRepPeak) state.currentRepPeak = v;

```
// Check for lockout
const lockoutOk =
  isAboveShoulder &&
  Math.abs(vyS) < CONFIG.LOCKOUT_VY_CUTOFF &&
  v < CONFIG.LOCKOUT_SPEED_CUTOFF;

if (lockoutOk) {
  state.overheadHoldCount++;
  if (state.overheadHoldCount >= CONFIG.OVERHEAD_HOLD_FRAMES) {
    finishRep();
  }
} else {
  state.overheadHoldCount = 0;
  // If dropped back to bottom without lockout, reset
  if (isBelowHip) {
    state.phase = "BOTTOM";
    state.currentRepPeak = 0;
    console.log("[PHASE] -> BOTTOM (failed rep)");
  }
}
```

}

state.prevWrist = { xNorm: wrist.x, yNorm: wrist.y, tMs: timeMs };
}

function finishRep() {
state.phase = “LOCKOUT”;
state.overheadHoldCount = 0;

state.repCount++;
state.currentSetReps++;
state.repHistory.push(state.currentRepPeak);

console.log(”[REP]”, state.repCount, “peak:”, state.currentRepPeak.toFixed(2), “m/s”);

document.getElementById(“val-reps”).textContent = String(state.repCount);
document.getElementById(“val-peak”).textContent = state.currentRepPeak.toFixed(2);

const dropEl = document.getElementById(“val-drop”);

if (state.repCount <= CONFIG.BASELINE_REPS) {
state.baseline = avg(state.repHistory);
dropEl.textContent = “CALC…”;
dropEl.style.color = “#94a3b8”;
} else {
const drop = (state.baseline - state.currentRepPeak) / state.baseline;
const dropPct = (drop * 100).toFixed(1);
dropEl.textContent = drop >= 0 ? `-${dropPct}%` : `+${Math.abs(dropPct)}%`;

```
if (drop * 100 >= CONFIG.DROP_FAIL) dropEl.style.color = "#ef4444";
else if (drop * 100 >= CONFIG.DROP_WARN) dropEl.style.color = "#fbbf24";
else dropEl.style.color = "#10b981";
```

}

// Reset for next rep
state.currentRepPeak = 0;
}

/* —————————— DRAWING —————————— */

function drawOverlay() {
state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
if (!state.lastPose) return;

if (state.isTestRunning && state.testStage === “ARMING”) {
drawFloorDebug(state.lastPose, “left”);
drawFloorDebug(state.lastPose, “right”);

```
const Lw = state.lastPose[CONFIG.LEFT.WRIST];
const Rw = state.lastPose[CONFIG.RIGHT.WRIST];
if (Lw) drawDot(Lw, state.floorMsLeft > 50, "#60a5fa");
if (Rw) drawDot(Rw, state.floorMsRight > 50, "#f87171");
return;
```

}

if (state.sideLocked && (state.lockedSide === “left” || state.lockedSide === “right”)) {
const idx = state.lockedSide === “left” ? CONFIG.LEFT : CONFIG.RIGHT;
const wrist = state.lastPose[idx.WRIST];
const shoulder = state.lastPose[idx.SHOULDER];
const hip = state.lastPose[idx.HIP];

```
if (shoulder && hip) drawZones(shoulder, hip, 0.65);
if (wrist) {
  const color = state.phase === "LOCKOUT" ? "#10b981" : 
                state.phase === "CONCENTRIC" ? "#fbbf24" : "#3b82f6";
  drawDot(wrist, true, color);
}
```

}
}

function drawFloorDebug(pose, side) {
const idx = side === “left” ? CONFIG.LEFT : CONFIG.RIGHT;
const knee = pose[idx.KNEE];
const ankle = pose[idx.ANKLE];
if (!knee || !ankle) return;

const shank = ankle.y - knee.y;
if (shank < CONFIG.MIN_SHANK_LEN_NORM) return;

const threshold = knee.y + CONFIG.SHANK_FRACTION * shank;
const yPx = threshold * state.canvas.height;

state.ctx.strokeStyle = side === “left” ? “rgba(96,165,250,0.5)” : “rgba(248,113,113,0.5)”;
state.ctx.lineWidth = 2;
state.ctx.setLineDash([5, 5]);
state.ctx.beginPath();
state.ctx.moveTo(0, yPx);
state.ctx.lineTo(state.canvas.width, yPx);
state.ctx.stroke();
state.ctx.setLineDash([]);
}

function drawZones(shoulder, hip, alpha = 0.6) {
const ctx = state.ctx;
const w = state.canvas.width;
const h = state.canvas.height;

// Shoulder line (green - lockout zone)
const shoulderY = shoulder.y * h;
ctx.strokeStyle = `rgba(16, 185, 129, ${alpha})`;
ctx.lineWidth = 2;
ctx.beginPath();
ctx.moveTo(0, shoulderY);
ctx.lineTo(w, shoulderY);
ctx.stroke();

ctx.fillStyle = `rgba(16, 185, 129, ${alpha})`;
ctx.font = “12px sans-serif”;
ctx.fillText(“LOCKOUT”, 5, shoulderY - 5);

// Hip line (blue - bottom zone)
const hipY = hip.y * h;
ctx.strokeStyle = `rgba(59, 130, 246, ${alpha})`;
ctx.beginPath();
ctx.moveTo(0, hipY);
ctx.lineTo(w, hipY);
ctx.stroke();

ctx.fillStyle = `rgba(59, 130, 246, ${alpha})`;
ctx.fillText(“HIP”, 5, hipY - 5);
}

function drawDot(wrist, emphasized = false, overrideColor = null) {
const x = wrist.x * state.canvas.width;
const y = wrist.y * state.canvas.height;

const ctx = state.ctx;
ctx.fillStyle = overrideColor || (emphasized ? “#10b981” : “#ef4444”);
ctx.beginPath();
ctx.arc(x, y, emphasized ? 12 : 8, 0, 2 * Math.PI);
ctx.fill();

ctx.fillStyle = “white”;
ctx.beginPath();
ctx.arc(x, y, 3, 0, 2 * Math.PI);
ctx.fill();
}

/* —————————— CONTROLS —————————— */

function setupControls() {
const startBtn = document.getElementById(“btn-start-test”);
const resetBtn = document.getElementById(“btn-reset”);

startBtn.onclick = async () => {
if (!state.isVideoReady) {
console.log(”[START] Video not ready”);
return;
}

```
console.log("[START] Beginning test");

state.isTestRunning = true;
state.testStage = "ARMING";

state.lockedSide = "unknown";
state.sideLocked = false;
state.lockedAtMs = 0;

state.floorMsLeft = 0;
state.floorMsRight = 0;
state.floorMsLocked = 0;
state.lastFloorUpdateMs = null;

resetSetMechanicsOnly();

startBtn.textContent = "Running…";
startBtn.disabled = true;
resetBtn.disabled = false;

setStatus("ARMING: Grab bell to lock side…", "#fbbf24");
dbg("dbg-stage", "ARMING");

try {
  await state.video.play();
  console.log("[START] Video playing");
} catch (e) {
  console.error("[START] Play failed:", e);
  alert("Playback blocked. Tap video area, then press Start again.");
  startBtn.textContent = "▶ Start Test";
  startBtn.disabled = false;
  resetBtn.disabled = true;
  state.isTestRunning = false;
  state.testStage = "IDLE";
}
```

};

resetBtn.onclick = () => {
console.log(”[RESET]”);
state.video.pause();
state.video.currentTime = 0;
resetAll(“Reset — ready”);
primeVideo();
};
}

/* —————————— RESET HELPERS —————————— */

function resetSetMechanicsOnly() {
state.prevWrist = null;
state.lockedCalibration = null;
state.smoothedVelocity = null;
state.smoothedVy = null;
state.lastSpeed = 0;
state.lastVy = 0;

state.phase = “IDLE”;
state.currentRepPeak = 0;
state.overheadHoldCount = 0;

document.getElementById(“val-velocity”).textContent = “0.00”;
document.getElementById(“val-peak”).textContent = “0.00”;
}

function resetAll(statusText) {
state.isTestRunning = false;
state.testStage = “IDLE”;

state.lockedSide = “unknown”;
state.sideLocked = false;
state.lockedAtMs = 0;

state.floorMsLeft = 0;
state.floorMsRight = 0;
state.floorMsLocked = 0;
state.lastFloorUpdateMs = null;

state.repCount = 0;
state.repHistory = [];
state.baseline = 0;
state.setCount = 0;
state.currentSetReps = 0;

resetSetMechanicsOnly();

document.getElementById(“val-reps”).textContent = “0”;
document.getElementById(“val-sets”).textContent = “0”;
const drop = document.getElementById(“val-drop”);
drop.textContent = “–”;
drop.style.color = “#f1f5f9”;

const startBtn = document.getElementById(“btn-start-test”);
const resetBtn = document.getElementById(“btn-reset”);
startBtn.textContent = “▶ Start Test”;
startBtn.disabled = false;
resetBtn.disabled = true;

setStatus(statusText, “#3b82f6”);
dbg(“dbg-stage”, “IDLE”);
dbg(“dbg-side”, “–”);
dbg(“dbg-phase”, “IDLE”);
}

/* —————————— UTILS —————————— */

function avg(arr) {
if (!arr.length) return 0;
return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function setStatus(text, color) {
const pill = document.getElementById(“status-pill”);
pill.textContent = text;
pill.style.color = color;
pill.style.borderColor = color;
}

// Start
initializeApp();