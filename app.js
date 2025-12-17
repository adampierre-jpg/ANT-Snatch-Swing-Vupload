/**

- Version 3.4 — FIXED HAND DETECTION TIMING
- - Fix: Active side detection now runs BEFORE physics
- - Fix: Head detection + lowest hand combined for reliable tracking
- - Maintains all debug features from v3.3
    */

import { PoseLandmarker, FilesetResolver } from “https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs”;

// ============================================
// CONFIG
// ============================================
const CONFIG = {
LEFT: {
WRIST: 15,
SHOULDER: 11,
HIP: 23,
KNEE: 25
},
RIGHT: {
WRIST: 16,
SHOULDER: 12,
HIP: 24,
KNEE: 26
},

HEAD_LANDMARK: 0,
TORSO_METERS: 0.45,

// Velocity Physics
SMOOTHING_ALPHA: 0.15,
MAX_REALISTIC_VELOCITY: 8.0,
ZERO_BAND: 0.1,
MIN_DT: 0.016,
MAX_DT: 0.1,

// Lockout Detection
LOCKOUT_VY_CUTOFF: 0.35,
LOCKOUT_SPEED_CUTOFF: 1.0,

// Session Logic
RESET_GRACE_MS_AFTER_LOCK: 2000,

// MediaPipe
MIN_DET_CONF: 0.5,
MIN_TRACK_CONF: 0.5,

// Thresholds (relaxed for debugging)
HEAD_DIP_THRESHOLD: 0.03,
HIKE_VY_THRESHOLD: 0.3,
HIKE_SPEED_THRESHOLD: 0.4,

// Head tracking for side confirmation
HEAD_DIP_CONFIRM_FRAMES: 3,

MAKE_WEBHOOK_URL: “https://hook.us2.make.com/bxyeuukaw4v71k32vx26jwiqbumgi19c”,

DEBUG_MODE: true
};

// ============================================
// STATE
// ============================================
let state = {
video: null,
canvas: null,
ctx: null,
landmarker: null,
isModelLoaded: false,
isVideoReady: false,

isTestRunning: false,
testStage: “IDLE”,
timeMs: 0,

lastPose: null,

// Hand tracking
activeTrackingSide: “left”,
lockedSide: “unknown”,
armingSide: null,

// Head tracking for side detection
headDipFrameCount: 0,
lastHeadY: 0,
candidateSide: null,

// Physics (used for active hand)
prevWrist: null,
lockedCalibration: null,
smoothedVelocity: 0,
smoothedVy: 0,
lastSpeed: 0,
lastVy: 0,

// Gesture Detection
parkingConfirmed: false,
prevHeadY: 0,

// Rep Logic
phase: “IDLE”,
currentRepPeak: 0,
overheadHoldCount: 0,

session: {
currentSet: null,
history: []
},

repHistory: []
};

// ============================================
// INITIALIZATION
// ============================================
async function initializeApp() {
state.video = document.getElementById(“video”);
state.canvas = document.getElementById(“canvas”);
state.ctx = state.canvas.getContext(“2d”);

document.getElementById(“btn-camera”).onclick = startCamera;
document.getElementById(“file-input”).onchange = handleUpload;
document.getElementById(“btn-start-test”).onclick = toggleTest;
document.getElementById(“btn-reset”).onclick = resetSession;
const saveBtn = document.getElementById(“btn-save”);
if (saveBtn) saveBtn.onclick = exportToMake;

const visionGen = await FilesetResolver.forVisionTasks(
“https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm”
);

state.landmarker = await PoseLandmarker.createFromOptions(visionGen, {
baseOptions: {
modelAssetPath: “https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task”,
delegate: “GPU”
},
runningMode: “VIDEO”,
numPoses: 1,
minPoseDetectionConfidence: CONFIG.MIN_DET_CONF,
minTrackingConfidence: CONFIG.MIN_TRACK_CONF
});

state.isModelLoaded = true;
document.getElementById(“loading-overlay”).classList.add(“hidden”);
setStatus(“Ready — Upload Video or Start Camera”, “#3b82f6”);

state.video.addEventListener(“loadeddata”, onVideoReady);
requestAnimationFrame(masterLoop);
}

// ============================================
// VIDEO INPUTS
// ============================================
function handleUpload(e) {
const file = e.target.files?.[0];
if (!file) return;

resetSession();
setStatus(“Loading Video…”, “#fbbf24”);

state.video.srcObject = null;
state.video.src = URL.createObjectURL(file);
state.video.load();
}

async function startCamera() {
resetSession();
const stream = await navigator.mediaDevices.getUserMedia({
video: { facingMode: “user”, width: { ideal: 1280 }, height: { ideal: 720 } },
audio: false
});
state.video.srcObject = stream;
state.video.src = “”;
state.video.play();
}

function onVideoReady() {
state.isVideoReady = true;
state.canvas.width = state.video.videoWidth;
state.canvas.height = state.video.videoHeight;
document.getElementById(“btn-start-test”).disabled = false;

if (state.video.src) {
const p = state.video.play();
if (p && typeof p.then === “function”) {
p.then(() => {
setTimeout(() => {
if (!state.isTestRunning) state.video.pause();
state.video.currentTime = 0;
}, 100);
}).catch(() => {});
}
}
setStatus(“Video Ready — Press Start Test”, “#3b82f6”);
}

async function toggleTest() {
if (!state.isTestRunning) {
state.isTestRunning = true;
state.testStage = “IDLE”;
document.getElementById(“btn-start-test”).textContent = “Stop Test”;
document.getElementById(“btn-reset”).disabled = false;
setStatus(“Scanning: Park hand below knee…”, “#fbbf24”);

```
if (state.video.paused) {
  try { await state.video.play(); } catch(e) {}
}
```

} else {
state.isTestRunning = false;
state.testStage = “IDLE”;
document.getElementById(“btn-start-test”).textContent = “Start Test”;
setStatus(“Stopped”, “#94a3b8”);
state.video.pause();
}
}

// ============================================
// MASTER LOOP
// ============================================
async function masterLoop(timestamp) {
requestAnimationFrame(masterLoop);

if (!state.isModelLoaded || !state.video) return;

state.timeMs = timestamp;

// Clear and draw video
state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
state.ctx.drawImage(state.video, 0, 0, state.canvas.width, state.canvas.height);

// Detect Pose
let pose = null;
if (state.landmarker && state.video.readyState >= 2) {
try {
const results = state.landmarker.detectForVideo(state.video, timestamp);
if (results && results.landmarks && results.landmarks.length > 0) {
pose = results.landmarks[0];
state.lastPose = pose;
}
} catch(e) {
console.warn(“Detection error:”, e);
}
}

if (!pose) {
drawOverlay();
return;
}

if (state.isTestRunning) {
// ✅ FIX: Detect active side BEFORE running physics
if (state.testStage === “IDLE”) {
detectActiveSide(pose);
}

```
// Now physics will use the correct side
runPhysics(pose, state.timeMs);

if (state.testStage === "IDLE") {
  checkStartCondition(pose, state.timeMs);
}

if (state.testStage === "RUNNING") {
  runSnatchLogic(pose);
  checkEndCondition(pose, state.timeMs);
}
```

}

drawOverlay();
}

// ============================================
// ACTIVE SIDE DETECTION (NEW FUNCTION)
// ============================================
function detectActiveSide(pose) {
const lWrist = pose[CONFIG.LEFT.WRIST];
const rWrist = pose[CONFIG.RIGHT.WRIST];
const head = pose[CONFIG.HEAD_LANDMARK];

if (!lWrist || !rWrist || !head) return;

// Determine which hand is lower (higher Y = lower on screen)
const lY = lWrist.y;
const rY = rWrist.y;
const lowestSide = lY > rY ? “left” : “right”;

// Check if head is dipping (indicates intentional hike position)
const headDipping = head.y > state.lastHeadY + CONFIG.HEAD_DIP_THRESHOLD;

// Check if lowest hand is in the floor zone
const inFloorZone = isWristInFloorZone(pose, lowestSide);

if (CONFIG.DEBUG_MODE) {
console.log(`[SIDE DETECT] Lowest:${lowestSide} | InZone:${inFloorZone} | HeadDip:${headDipping} | HeadY:${head.y.toFixed(3)} | LastHeadY:${state.lastHeadY.toFixed(3)}`);
}

// If hand is in floor zone AND head is dipping, this confirms intent
if (inFloorZone && headDipping) {
if (state.candidateSide === lowestSide) {
state.headDipFrameCount++;
} else {
state.candidateSide = lowestSide;
state.headDipFrameCount = 1;
}

```
// Require multiple frames of confirmation to avoid false triggers
if (state.headDipFrameCount >= CONFIG.HEAD_DIP_CONFIRM_FRAMES) {
  state.activeTrackingSide = lowestSide;
  if (CONFIG.DEBUG_MODE) {
    console.log(`✅ ACTIVE SIDE CONFIRMED: ${lowestSide} (head dip + floor zone)`);
  }
}
```

} else if (inFloorZone) {
// Hand in zone but no head dip - still track it but don’t fully confirm
state.activeTrackingSide = lowestSide;
} else {
// Reset confirmation if conditions aren’t met
state.headDipFrameCount = 0;
state.candidateSide = null;
}

// Update head tracking
state.lastHeadY = head.y;
}

// ============================================
// START/END CONDITIONS
// ============================================
function checkStartCondition(pose, timeMs) {
if (state.testStage !== “IDLE”) return;

const head = pose[CONFIG.HEAD_LANDMARK];
if (!head) return;

// Use the already-detected active side
const activeSide = state.activeTrackingSide;
const sideIdx = activeSide === “left” ? CONFIG.LEFT : CONFIG.RIGHT;
const wrist = pose[sideIdx.WRIST];

if (!wrist) return;

const inZone = isWristInFloorZone(pose, activeSide);
const headLowering = head.y > state.prevHeadY + CONFIG.HEAD_DIP_THRESHOLD;
const hikingDown = state.lastVy > CONFIG.HIKE_VY_THRESHOLD && state.lastSpeed > CONFIG.HIKE_SPEED_THRESHOLD;

if (CONFIG.DEBUG_MODE && inZone) {
console.log(`[START CHECK] Side:${activeSide} | Zone:${inZone} | HeadDip:${headLowering} | Hike:${hikingDown} | Vy:${state.lastVy.toFixed(2)} | Speed:${state.lastSpeed.toFixed(2)}`);
}

state.prevHeadY = head.y;

// Park confirm: requires head dip while in zone
if (inZone && headLowering) {
state.parkingConfirmed = true;
state.armingSide = activeSide;
if (CONFIG.DEBUG_MODE) console.log(`✅ PARKING CONFIRMED: ${activeSide}`);
}

// Start trigger: parked + hiking motion detected
if (state.parkingConfirmed && inZone && hikingDown) {
if (CONFIG.DEBUG_MODE) console.log(`🚀 STARTING SET: ${state.armingSide}`);
startNewSet(state.armingSide);
state.parkingConfirmed = false;
state.prevHeadY = 0;
}
}

function checkEndCondition(pose, timeMs) {
if (state.testStage !== “RUNNING”) return;
if (!state.session.currentSet) return;

const grace = timeMs - state.session.currentSet.lockedAtMs;
if (grace < CONFIG.RESET_GRACE_MS_AFTER_LOCK) return;

const sideIdx = state.lockedSide === “left” ? CONFIG.LEFT : CONFIG.RIGHT;
const wrist = pose[sideIdx.WRIST];
const head = pose[CONFIG.HEAD_LANDMARK];
if (!wrist || !head) return;

const inZone = isWristInFloorZone(pose, state.lockedSide);
const headLowering = head.y > state.prevHeadY + CONFIG.HEAD_DIP_THRESHOLD;
const standingUp = state.lastVy < -0.3 && state.lastSpeed > 0.4;

if (CONFIG.DEBUG_MODE && inZone) {
console.log(`[END CHECK] Zone:${inZone} | HeadDip:${headLowering} | StandUp:${standingUp} | Vy:${state.lastVy.toFixed(2)}`);
}

state.prevHeadY = head.y;

// Park confirm for ending
if (inZone && headLowering) {
state.parkingConfirmed = true;
if (CONFIG.DEBUG_MODE) console.log(`✅ PARKING CONFIRMED (END)`);
}

// End trigger: parked + standing up motion
if (state.parkingConfirmed && inZone && standingUp) {
if (CONFIG.DEBUG_MODE) console.log(`🛑 ENDING SET`);
endCurrentSet();
state.parkingConfirmed = false;
state.prevHeadY = 0;
}
}

// ============================================
// SET MANAGEMENT
// ============================================
function startNewSet(side) {
state.lockedSide = side;
state.testStage = “RUNNING”;

// Reset tracking
state.repHistory = [];
state.currentRepPeak = 0;
state.overheadHoldCount = 0;
state.phase = “IDLE”;
state.lockedCalibration = null;
state.prevWrist = null;

// Reset head tracking for side detection
state.headDipFrameCount = 0;
state.candidateSide = null;

state.session.currentSet = {
id: state.session.history.length + 1,
hand: side,
reps: [],
startTime: new Date(),
lockedAtMs: state.timeMs
};

updateUIValues(0, 0);
setStatus(`LOCKED: ${side.toUpperCase()}`, “#10b981”);
}

function endCurrentSet() {
if (state.session.currentSet) {
state.session.currentSet.endTime = new Date();

```
const peaks = state.session.currentSet.reps;
const avg = peaks.length > 0 ? peaks.reduce((a,b)=>a+b,0)/peaks.length : 0;
state.session.currentSet.avgVelocity = avg.toFixed(2);

state.session.history.push(state.session.currentSet);
```

}

state.testStage = “IDLE”;
state.lockedSide = “unknown”;
state.session.currentSet = null;
state.activeTrackingSide = “left”;

// Reset head tracking
state.headDipFrameCount = 0;
state.candidateSide = null;
state.lastHeadY = 0;

setStatus(“Set Saved. Park to start next.”, “#3b82f6”);
}

// ============================================
// PHYSICS ENGINE
// ============================================
function runPhysics(pose, timeMs) {
// In IDLE: Track the detected active side (already set by detectActiveSide)
// In RUNNING: Track the locked side
const side = state.testStage === “IDLE” ? state.activeTrackingSide : state.lockedSide;
if (!side || side === “unknown”) return;

const idx = side === “left” ? CONFIG.LEFT : CONFIG.RIGHT;
const wrist = pose[idx.WRIST];
const shoulder = pose[idx.SHOULDER];
const hip = pose[idx.HIP];
if (!wrist || !shoulder || !hip) return;

// Calibration (torso length in pixels to meters conversion)
if (!state.lockedCalibration) {
const torsoPx = Math.abs(shoulder.y - hip.y) * state.canvas.height;
state.lockedCalibration = Math.max(50, torsoPx / CONFIG.TORSO_METERS);
}

if (!state.prevWrist) {
state.prevWrist = { x: wrist.x, y: wrist.y, t: timeMs };
return;
}

const dt = (timeMs - state.prevWrist.t) / 1000;
if (dt < CONFIG.MIN_DT || dt > CONFIG.MAX_DT) {
state.prevWrist = { x: wrist.x, y: wrist.y, t: timeMs };
return;
}

const dxPx = (wrist.x - state.prevWrist.x) * state.canvas.width;
const dyPx = (wrist.y - state.prevWrist.y) * state.canvas.height;

let vx = (dxPx / state.lockedCalibration) / dt;
let vy = (dyPx / state.lockedCalibration) / dt;
let speed = Math.hypot(vx, vy);

// Frame normalization to 30fps baseline
const TARGET_FPS = 30;
const frameTimeMs = 1000 / TARGET_FPS;
const actualFrameTimeMs = timeMs - state.prevWrist.t;
const timeRatio = frameTimeMs / actualFrameTimeMs;
vx *= timeRatio;
vy *= timeRatio;
speed = Math.hypot(vx, vy);

// Zero band (ignore micro-movements)
if (speed < CONFIG.ZERO_BAND) speed = 0;

// Exponential smoothing
state.smoothedVelocity = CONFIG.SMOOTHING_ALPHA * speed + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVelocity;
state.smoothedVy = CONFIG.SMOOTHING_ALPHA * vy + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVy;

// Apply ceiling
state.lastSpeed = Math.min(state.smoothedVelocity, CONFIG.MAX_REALISTIC_VELOCITY);
state.lastVy = Math.min(Math.max(state.smoothedVy, -CONFIG.MAX_REALISTIC_VELOCITY), CONFIG.MAX_REALISTIC_VELOCITY);

state.prevWrist = { x: wrist.x, y: wrist.y, t: timeMs };

// Update UI velocity display
if (state.testStage === “RUNNING”) {
document.getElementById(“val-velocity”).textContent = state.lastSpeed.toFixed(2);
}
}

// ============================================
// REP DETECTION
// ============================================
function runSnatchLogic(pose) {
const v = state.smoothedVelocity;
const vy = state.smoothedVy;

const idx = state.lockedSide === “left” ? CONFIG.LEFT : CONFIG.RIGHT;
const wrist = pose[idx.WRIST];
const hip = pose[idx.HIP];
const shoulder = pose[idx.SHOULDER];
const nose = pose[CONFIG.HEAD_LANDMARK];

if (!wrist || !hip || !shoulder || !nose) return;

const isBelowHip = wrist.y > hip.y;
const isAboveShoulder = wrist.y < shoulder.y;
const isAboveNose = wrist.y < nose.y;

if (state.phase === “IDLE” || state.phase === “LOCKOUT”) {
if (isBelowHip) {
state.phase = “BOTTOM”;
state.overheadHoldCount = 0;
}
}
else if (state.phase === “BOTTOM”) {
if (!isBelowHip) {
state.phase = “CONCENTRIC”;
state.currentRepPeak = 0;
}
}
else if (state.phase === “CONCENTRIC”) {
// Track peak velocity until shoulder height
if (!isAboveShoulder) {
if (v > state.currentRepPeak) state.currentRepPeak = v;
}

```
// Check for stable lockout position
const isStable = Math.abs(vy) < CONFIG.LOCKOUT_VY_CUTOFF && v < CONFIG.LOCKOUT_SPEED_CUTOFF;

if (isAboveNose && isStable) {
  state.overheadHoldCount++;
  if (state.overheadHoldCount >= 2) {
    recordRep();
  }
}
```

}
}

function recordRep() {
state.phase = “LOCKOUT”;
state.overheadHoldCount = 0;

if (state.session.currentSet) {
state.session.currentSet.reps.push(state.currentRepPeak);
}

state.repHistory.push(state.currentRepPeak);
updateUIValues(state.repHistory.length, state.currentRepPeak);

if (CONFIG.DEBUG_MODE) console.log(`📊 REP RECORDED: ${state.currentRepPeak.toFixed(2)} m/s`);
}

// ============================================
// HELPERS
// ============================================
function isWristInFloorZone(pose, side) {
const idx = side === “left” ? CONFIG.LEFT : CONFIG.RIGHT;
const wrist = pose[idx.WRIST];
const knee = pose[idx.KNEE];

if (!wrist || !knee) return false;
return wrist.y > knee.y;
}

function resetSession() {
state.session = { currentSet: null, history: [] };
state.repHistory = [];
state.testStage = “IDLE”;
state.lockedSide = “unknown”;
state.activeTrackingSide = “left”;
state.armingSide = null;
state.smoothedVelocity = 0;
state.smoothedVy = 0;
state.lastSpeed = 0;
state.lastVy = 0;
state.lockedCalibration = null;
state.prevWrist = null;
state.parkingConfirmed = false;
state.prevHeadY = 0;

// Reset head tracking for side detection
state.headDipFrameCount = 0;
state.candidateSide = null;
state.lastHeadY = 0;

updateUIValues(0, 0);
setStatus(“Session Cleared — Ready”, “#3b82f6”);
}

// ============================================
// UI
// ============================================
function updateUIValues(reps, peak) {
document.getElementById(“val-reps”).textContent = reps;
document.getElementById(“val-peak”).textContent = peak.toFixed(2);
}

function setStatus(text, color) {
const pill = document.getElementById(“status-pill”);
if (pill) {
pill.textContent = text;
pill.style.color = color;
pill.style.borderColor = color;
}
}

function drawOverlay() {
if (!state.lastPose) return;

// Debug overlay (top-left corner)
if (CONFIG.DEBUG_MODE) {
state.ctx.fillStyle = “#fbbf24”;
state.ctx.font = “12px monospace”;
state.ctx.fillText(`Side: ${state.testStage === "IDLE" ? state.activeTrackingSide : state.lockedSide}`, 10, 20);
state.ctx.fillText(`Speed: ${state.lastSpeed.toFixed(2)} m/s`, 10, 35);
state.ctx.fillText(`Vy: ${state.lastVy.toFixed(2)} m/s`, 10, 50);
state.ctx.fillText(`Stage: ${state.testStage}`, 10, 65);
state.ctx.fillText(`Parked: ${state.parkingConfirmed}`, 10, 80);
state.ctx.fillText(`HeadDipFrames: ${state.headDipFrameCount}`, 10, 95);
state.ctx.fillText(`Candidate: ${state.candidateSide || "none"}`, 10, 110);
}

const side = state.testStage === “IDLE” ? state.activeTrackingSide : state.lockedSide;

if (state.testStage === “RUNNING” && side !== “unknown”) {
const idx = side === “left” ? CONFIG.LEFT : CONFIG.RIGHT;
const wrist = state.lastPose[idx.WRIST];
drawDot(wrist, true, “#10b981”);

```
const inZone = isWristInFloorZone(state.lastPose, side);
drawParkingLine(state.lastPose, side, inZone);
```

} else {
// In IDLE: show both wrists, highlight the lowest
const lWrist = state.lastPose[CONFIG.LEFT.WRIST];
const rWrist = state.lastPose[CONFIG.RIGHT.WRIST];
const lY = lWrist?.y || 0;
const rY = rWrist?.y || 0;
const lowest = lY > rY ? “left” : “right”;

```
// Color based on confirmation status
const confirmedColor = "#10b981";  // Green when head dip confirmed
const pendingColor = "#fbbf24";    // Yellow when pending
const color = state.headDipFrameCount >= CONFIG.HEAD_DIP_CONFIRM_FRAMES ? confirmedColor : pendingColor;

drawDot(lWrist, lowest === "left", lowest === "left" ? color : "#94a3b8");
drawDot(rWrist, lowest === "right", lowest === "right" ? color : "#94a3b8");

drawParkingLine(state.lastPose, lowest, isWristInFloorZone(state.lastPose, lowest));
```

}
}

function drawParkingLine(pose, side, isActive) {
const idx = side === “left” ? CONFIG.LEFT : CONFIG.RIGHT;
const knee = pose[idx.KNEE];
if (!knee) return;

const y = knee.y * state.canvas.height;
state.ctx.beginPath();
state.ctx.strokeStyle = isActive ? “#10b981” : “rgba(255,255,255,0.2)”;
state.ctx.lineWidth = 2;
state.ctx.moveTo(0, y);
state.ctx.lineTo(state.canvas.width, y);
state.ctx.stroke();
}

function drawDot(landmark, big, color) {
if (!landmark) return;
const x = landmark.x * state.canvas.width;
const y = landmark.y * state.canvas.height;
state.ctx.beginPath();
state.ctx.fillStyle = color;
state.ctx.arc(x, y, big ? 8 : 5, 0, 2*Math.PI);
state.ctx.fill();
}

// ============================================
// EXPORT
// ============================================
async function exportToMake() {
const history = state.session.history;

if (!history.length) {
alert(“No completed sets to export.”);
return;
}

const totalReps = history.reduce((sum, set) => sum + set.reps.length, 0);
const totalAvg = history.length > 0 ?
(history.reduce((sum, set) => sum + parseFloat(set.avgVelocity), 0) / history.length) : 0;

const payload = {
athlete_id: “dad_ready_user”,
session_date: new Date().toISOString(),
total_reps: totalReps,
session_avg_velocity: totalAvg.toFixed(2),
sets: history.map((set, index) => ({
set_order: index + 1,
hand: set.hand,
rep_count: set.reps.length,
peak_velocity_avg: parseFloat(set.avgVelocity),
raw_peaks: set.reps
}))
};

console.log(“EXPORTING TO MAKE:”, JSON.stringify(payload, null, 2));
setStatus(“Exporting to Make…”, “#8b5cf6”);

try {
const response = await fetch(CONFIG.MAKE_WEBHOOK_URL, {
method: “POST”,
headers: { “Content-Type”: “application/json” },
body: JSON.stringify(payload)
});

```
if(response.ok) {
  setStatus("Success! Session Saved.", "#10b981");
} else {
  setStatus("Error: Make.com rejected payload.", "#ef4444");
}
```

} catch(e) {
console.error(“Network Error:”, e);
setStatus(“Network Error (Check Console)”, “#ef4444”);
}
}

initializeApp();