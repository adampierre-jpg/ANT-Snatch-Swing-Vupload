/**
 * KB Velocity Tracker (Snatch) — Hike-pass “hand on bell” handedness
 *
 * What changed vs the previous JS:
 * - Handedness is determined at Start Test by the FIRST wrist that enters a “floor zone”
 *   (wrist significantly below its knee, scaled by shank length knee→ankle).
 * - When the bell is set down (locked wrist returns to floor zone and DWELLS), handedness resets
 *   so the next set can be auto-detected again.
 * - Everything else (primeVideo, masterLoop, snatch phase logic, smoothing, calibration, drop-off) stays consistent.
 *
 * Landmark indices used include:
 * - Left wrist 15 / right wrist 16
 * - Left knee 25 / right knee 26
 * - Left ankle 27 / right ankle 28  [web:54][web:53]
 */

import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const CONFIG = {
  // Pose indices
  LEFT:  { WRIST: 15, SHOULDER: 11, HIP: 23, KNEE: 25, ANKLE: 27 },
  RIGHT: { WRIST: 16, SHOULDER: 12, HIP: 24, KNEE: 26, ANKLE: 28 },

  // MediaPipe
  MIN_DET_CONF: 0.5,
  MIN_TRACK_CONF: 0.5,

  // Velocity stabilizers
  SMOOTHING_ALPHA: 0.30,
  MAX_REALISTIC_VELOCITY: 10.0,
  MIN_DT: 0.01,
  MAX_DT: 0.10,
  TORSO_METERS: 0.5,

  // Snatch rep logic
  LOCKOUT_VEL_CUTOFF: 0.5,
  BASELINE_REPS: 3,
  DROP_WARN: 15,
  DROP_FAIL: 20,

  // Floor-zone handedness detection
  // wrist.y > knee.y + (SHANK_FRACTION * (ankle.y - knee.y))
  // higher fraction -> deeper below knee required (reduces false triggers)
  SHANK_FRACTION: 0.35,

  // Require consecutive frames “in zone” to confirm.
  ARM_FRAMES_REQUIRED: 5,     // ~80–150ms at typical mobile fps
  RESET_FRAMES_REQUIRED: 18,  // longer dwell for set-down (~300–600ms)

  // Additional guard to avoid random false positives
  MIN_SHANK_LEN_NORM: 0.06,   // if knee/ankle are too close (bad pose), ignore
};

let state = {
  // DOM / MediaPipe
  video: null,
  canvas: null,
  ctx: null,
  landmarker: null,

  // Flags
  isModelLoaded: false,
  isVideoReady: false,

  // Pose cache
  lastPose: null,

  // Test
  isTestRunning: false,
  testStage: "IDLE",            // IDLE | ARMING | RUNNING

  // Handedness
  lockedSide: "unknown",        // left | right | unknown
  sideLocked: false,

  // Floor-zone counters
  floorCountLeft: 0,
  floorCountRight: 0,
  floorCountLocked: 0,

  // Physics (locked side only)
  prevWrist: null,              // {xNorm, yNorm, tMs}
  lockedCalibration: null,      // px/m
  smoothedVelocity: 0,

  // Rep state
  phase: "IDLE",                // IDLE -> BOTTOM -> CONCENTRIC -> LOCKOUT
  repCount: 0,
  currentRepPeak: 0,
  repHistory: [],
  baseline: 0
};

/* ----------------------------- INIT ----------------------------- */

async function initializeApp() {
  try {
    state.video = document.getElementById("video");
    state.canvas = document.getElementById("canvas");
    state.ctx = state.canvas.getContext("2d");

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
    document.getElementById("loading-overlay").classList.add("hidden");
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
  } catch (e) {
    console.error(e);
    alert("Startup Error: " + e.message);
  }
}

/* -------------------------- SOURCE HANDLING -------------------------- */

function handleUpload(e) {
  const file = e.target.files?.[0];
  if (!file) return;

  resetAll("Loading…");
  state.video.srcObject = null;
  state.video.src = URL.createObjectURL(file);
  state.video.load();
}

async function startCamera() {
  try {
    resetAll("Starting camera…");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false
    });
    state.video.srcObject = stream;
    state.video.src = "";
  } catch (e) {
    alert("Camera Error: " + e.message);
  }
}

function onVideoReady() {
  state.isVideoReady = true;

  state.canvas.width = state.video.videoWidth || state.canvas.width;
  state.canvas.height = state.video.videoHeight || state.canvas.height;

  document.getElementById("btn-start-test").disabled = false;

  primeVideo();
  setStatus("Video Loaded — press Start Test", "#fbbf24");
}

function primeVideo() {
  const p = state.video.play();
  if (p && typeof p.then === "function") {
    p.then(() => {
      setTimeout(() => {
        if (!state.isTestRunning) state.video.pause();
        state.video.currentTime = 0;
      }, 120);
    }).catch(() => {
      console.log("Autoplay blocked.");
    });
  }
}

/* ---------------------------- MASTER LOOP ---------------------------- */

function masterLoop() {
  if (state.isModelLoaded && state.isVideoReady && state.video.readyState >= 2) {
    // monotonic timestamp for MediaPipe stability across seeks/resets
    const result = state.landmarker.detectForVideo(state.video, performance.now());

    if (result.landmarks && result.landmarks.length > 0) {
      state.lastPose = result.landmarks[0];

      // Stage logic:
      // ARMING: decide lockedSide via floor-zone entry
      if (state.isTestRunning && state.testStage === "ARMING") {
        updateHandednessByFloorGrab(state.lastPose);
      }

      // RUNNING: track snatch only on locked side; also watch for set-down to reset
      if (state.isTestRunning && state.testStage === "RUNNING" && !state.video.paused) {
        runSnatchPhysicsAndLogic(state.lastPose, state.video.currentTime * 1000);
        maybeResetOnSetDown(state.lastPose);
      }
    }
  }

  drawOverlay();
  requestAnimationFrame(masterLoop);
}

/* ----------------- FLOOR ZONE: ARMING + RESET ----------------- */

function updateHandednessByFloorGrab(pose) {
  // Compute floor-zone membership for each side
  const inFloorLeft = isWristInFloorZone(pose, "left");
  const inFloorRight = isWristInFloorZone(pose, "right");

  // Update consecutive counters
  state.floorCountLeft = inFloorLeft ? state.floorCountLeft + 1 : 0;
  state.floorCountRight = inFloorRight ? state.floorCountRight + 1 : 0;

  // Winner: first side to reach required dwell frames
  if (state.floorCountLeft >= CONFIG.ARM_FRAMES_REQUIRED && state.floorCountRight < CONFIG.ARM_FRAMES_REQUIRED) {
    lockSideAndStartRunning("left");
    return;
  }
  if (state.floorCountRight >= CONFIG.ARM_FRAMES_REQUIRED && state.floorCountLeft < CONFIG.ARM_FRAMES_REQUIRED) {
    lockSideAndStartRunning("right");
    return;
  }

  // If both hit at once, pick the one that is deeper (wrist.y - threshold bigger)
  if (state.floorCountLeft >= CONFIG.ARM_FRAMES_REQUIRED && state.floorCountRight >= CONFIG.ARM_FRAMES_REQUIRED) {
    const depthL = floorDepth(pose, "left");
    const depthR = floorDepth(pose, "right");
    lockSideAndStartRunning(depthL >= depthR ? "left" : "right");
  }
}

function lockSideAndStartRunning(side) {
  state.lockedSide = side;
  state.sideLocked = true;

  // The “grab” phase should not contaminate the rep 
  // reset physics + reps now so tracking starts clean AFTER the grab.
  resetRunStateOnly();

  state.testStage = "RUNNING";
  setStatus(`Locked ${side.toUpperCase()} — tracking`, "#10b981");
}

function maybeResetOnSetDown(pose) {
  if (!state.sideLocked) return;

  // If the locked wrist is in floor-zone long enough, treat as "bell set down"
  const inFloorLocked = isWristInFloorZone(pose, state.lockedSide);
  state.floorCountLocked = inFloorLocked ? state.floorCountLocked + 1 : 0;

  if (state.floorCountLocked >= CONFIG.RESET_FRAMES_REQUIRED) {
    // Reset handedness but keep video/test running (so next set can be detected)
    state.sideLocked = false;
    state.lockedSide = "unknown";
    state.testStage = "ARMING";

    // Clear counters + run state (ready for next set)
    state.floorCountLeft = 0;
    state.floorCountRight = 0;
    state.floorCountLocked = 0;

    resetRunStateOnly();
    setStatus("Set down detected — arm next set by grabbing bell", "#fbbf24");
  }
}

function isWristInFloorZone(pose, side) {
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;

  const wrist = pose[idx.WRIST];
  const knee = pose[idx.KNEE];
  const ankle = pose[idx.ANKLE];

  if (!wrist || !knee || !ankle) return false;

  // Need a meaningful shank length; if pose is garbage, ignore
  const shank = ankle.y - knee.y; // normalized (positive if ankle lower than knee)
  if (shank < CONFIG.MIN_SHANK_LEN_NORM) return false;

  const threshold = knee.y + CONFIG.SHANK_FRACTION * shank;

  // normalized y: larger means lower in image
  return wrist.y > threshold;
}

// Positive means “deeper into floor zone”
function floorDepth(pose, side) {
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;

  const wrist = pose[idx.WRIST];
  const knee = pose[idx.KNEE];
  const ankle = pose[idx.ANKLE];

  if (!wrist || !knee || !ankle) return 0;

  const shank = ankle.y - knee.y;
  if (shank < CONFIG.MIN_SHANK_LEN_NORM) return 0;

  const threshold = knee.y + CONFIG.SHANK_FRACTION * shank;
  return Math.max(0, wrist.y - threshold);
}

/* ---------------------- SNATCH PHYSICS + LOGIC ---------------------- */

function runSnatchPhysicsAndLogic(pose, timeMs) {
  const side = state.lockedSide;
  if (side !== "left" && side !== "right") return;

  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;

  const wrist = pose[idx.WRIST];
  const shoulder = pose[idx.SHOULDER];
  const hip = pose[idx.HIP];

  if (!wrist || !shoulder || !hip) return;
  if ((wrist.visibility ?? 1) < 0.5) return;

  // lock calibration once
  if (!state.lockedCalibration) {
    const torsoPx = Math.abs(shoulder.y - hip.y) * state.canvas.height;
    state.lockedCalibration = torsoPx > 0 ? torsoPx / CONFIG.TORSO_METERS : 100;
  }

  if (!state.prevWrist) {
    state.prevWrist = { xNorm: wrist.x, yNorm: wrist.y, tMs: timeMs };
    return;
  }

  const dt = (timeMs - state.prevWrist.tMs) / 1000;
  if (dt < CONFIG.MIN_DT || dt > CONFIG.MAX_DT) {
    state.prevWrist = { xNorm: wrist.x, yNorm: wrist.y, tMs: timeMs };
    return;
  }

  const dxPx = (wrist.x - state.prevWrist.xNorm) * state.canvas.width;
  const dyPx = (wrist.y - state.prevWrist.yNorm) * state.canvas.height;
  const distPx = Math.hypot(dxPx, dyPx);

  const rawVel = (distPx / state.lockedCalibration) / dt;
  if (rawVel > CONFIG.MAX_REALISTIC_VELOCITY) {
    state.prevWrist = { xNorm: wrist.x, yNorm: wrist.y, tMs: timeMs };
    return;
  }

  if (!state.smoothedVelocity) state.smoothedVelocity = rawVel;
  state.smoothedVelocity =
    CONFIG.SMOOTHING_ALPHA * rawVel + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVelocity;

  const velocity = state.smoothedVelocity;
  document.getElementById("val-velocity").textContent = velocity.toFixed(2);

  const isBelowHip = wrist.y > hip.y;
  const isAboveShoulder = wrist.y < shoulder.y;

  if (state.phase === "IDLE" || state.phase === "LOCKOUT") {
    if (isBelowHip) state.phase = "BOTTOM";
  } else if (state.phase === "BOTTOM") {
    if (wrist.y < hip.y) {
      state.phase = "CONCENTRIC";
      state.currentRepPeak = 0;
    }
  } else if (state.phase === "CONCENTRIC") {
    if (velocity > state.currentRepPeak) state.currentRepPeak = velocity;

    if (isAboveShoulder && velocity < CONFIG.LOCKOUT_VEL_CUTOFF) finishRep();
    else if (isBelowHip) state.phase = "BOTTOM";
  }

  state.prevWrist = { xNorm: wrist.x, yNorm: wrist.y, tMs: timeMs };
}

function finishRep() {
  state.phase = "LOCKOUT";
  state.repCount++;
  state.repHistory.push(state.currentRepPeak);

  document.getElementById("val-reps").textContent = String(state.repCount);
  document.getElementById("val-peak").textContent = state.currentRepPeak.toFixed(2);

  const dropEl = document.getElementById("val-drop");

  if (state.repCount <= CONFIG.BASELINE_REPS) {
    state.baseline = avg(state.repHistory);
    dropEl.textContent = "CALC...";
    dropEl.style.color = "#94a3b8";
  } else {
    const drop = (state.baseline - state.currentRepPeak) / state.baseline;
    const dropPct = (drop * 100).toFixed(1);
    dropEl.textContent = `-${dropPct}%`;

    if (drop * 100 >= CONFIG.DROP_FAIL) dropEl.style.color = "#ef4444";
    else if (drop * 100 >= CONFIG.DROP_WARN) dropEl.style.color = "#fbbf24";
    else dropEl.style.color = "#10b981";
  }
}

/* ------------------------------ DRAWING ------------------------------ */

function drawOverlay() {
  state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
  if (!state.lastPose) return;

  // While arming: draw both wrist markers and knee lines faintly for debugging
  if (state.isTestRunning && state.testStage === "ARMING") {
    drawFloorDebug(state.lastPose, "left");
    drawFloorDebug(state.lastPose, "right");

    const Lw = state.lastPose[CONFIG.LEFT.WRIST];
    const Rw = state.lastPose[CONFIG.RIGHT.WRIST];
    if (Lw) drawDot(Lw, false, "#60a5fa");
    if (Rw) drawDot(Rw, false, "#f87171");
    return;
  }

  // Running: draw zones + dot for locked side
  if (state.sideLocked && (state.lockedSide === "left" || state.lockedSide === "right")) {
    const idx = state.lockedSide === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
    const wrist = state.lastPose[idx.WRIST];
    const shoulder = state.lastPose[idx.SHOULDER];
    const hip = state.lastPose[idx.HIP];

    if (shoulder && hip) drawZones(shoulder, hip, 0.65);
    if (wrist) drawDot(wrist, true);
  }
}

function drawFloorDebug(pose, side) {
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const knee = pose[idx.KNEE];
  const ankle = pose[idx.ANKLE];
  if (!knee || !ankle) return;

  const shank = ankle.y - knee.y;
  if (shank < CONFIG.MIN_SHANK_LEN_NORM) return;

  const threshold = knee.y + CONFIG.SHANK_FRACTION * shank;
  const yPx = threshold * state.canvas.height;

  state.ctx.strokeStyle = side === "left" ? "rgba(96,165,250,0.35)" : "rgba(248,113,113,0.35)";
  state.ctx.lineWidth = 2;
  state.ctx.beginPath();
  state.ctx.moveTo(0, yPx);
  state.ctx.lineTo(state.canvas.width, yPx);
  state.ctx.stroke();
}

function drawZones(shoulder, hip, alpha = 0.6) {
  const ctx = state.ctx;
  const w = state.canvas.width;
  const h = state.canvas.height;

  const shoulderY = shoulder.y * h;
  ctx.strokeStyle = `rgba(16, 185, 129, ${alpha})`;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, shoulderY);
  ctx.lineTo(w, shoulderY);
  ctx.stroke();

  const hipY = hip.y * h;
  ctx.strokeStyle = `rgba(59, 130, 246, ${alpha})`;
  ctx.beginPath();
  ctx.moveTo(0, hipY);
  ctx.lineTo(w, hipY);
  ctx.stroke();
}

function drawDot(wrist, emphasized = false, overrideColor = null) {
  const x = wrist.x * state.canvas.width;
  const y = wrist.y * state.canvas.height;

  const ctx = state.ctx;
  ctx.fillStyle = overrideColor || (emphasized ? "#10b981" : "#ef4444");
  ctx.beginPath();
  ctx.arc(x, y, emphasized ? 10 : 7, 0, 2 * Math.PI);
  ctx.fill();

  ctx.fillStyle = "white";
  ctx.beginPath();
  ctx.arc(x, y, 3, 0, 2 * Math.PI);
  ctx.fill();
}

/* ------------------------------ CONTROLS ------------------------------ */

function setupControls() {
  const startBtn = document.getElementById("btn-start-test");
  const resetBtn = document.getElementById("btn-reset");

  startBtn.onclick = async () => {
    if (!state.isVideoReady) return;

    state.isTestRunning = true;
    state.testStage = "ARMING";

    // clear handedness + counters
    state.lockedSide = "unknown";
    state.sideLocked = false;
    state.floorCountLeft = 0;
    state.floorCountRight = 0;
    state.floorCountLocked = 0;

    // clear run state so after arming we start clean
    resetRunStateOnly();

    startBtn.textContent = "Test Running…";
    startBtn.disabled = true;
    resetBtn.disabled = false;

    setStatus("ARMING: grab bell (wrist deep below knee) to lock side…", "#fbbf24");

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
  };

  resetBtn.onclick = () => {
    state.video.pause();
    state.video.currentTime = 0;
    resetAll("Reset — ready");
    primeVideo();
  };
}

/* ------------------------------ RESET ------------------------------ */

function resetRunStateOnly() {
  state.prevWrist = null;
  state.lockedCalibration = null;
  state.smoothedVelocity = 0;

  state.phase = "IDLE";
  state.repCount = 0;
  state.currentRepPeak = 0;
  state.repHistory = [];
  state.baseline = 0;

  document.getElementById("val-velocity").textContent = "0.00";
  document.getElementById("val-peak").textContent = "0.00";
  document.getElementById("val-reps").textContent = "0";
  const drop = document.getElementById("val-drop");
  drop.textContent = "--";
  drop.style.color = "#f1f5f9";
}

function resetAll(statusText) {
  state.isTestRunning = false;
  state.testStage = "IDLE";

  state.lockedSide = "unknown";
  state.sideLocked = false;

  state.floorCountLeft = 0;
  state.floorCountRight = 0;
  state.floorCountLocked = 0;

  resetRunStateOnly();

  const startBtn = document.getElementById("btn-start-test");
  const resetBtn = document.getElementById("btn-reset");
  startBtn.textContent = "▶ Start Test";
  startBtn.disabled = false;
  resetBtn.disabled = true;

  setStatus(statusText, "#3b82f6");
}

/* ------------------------------ UTILS ------------------------------ */

function avg(arr) {
  if (!arr.length) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function setStatus(text, color) {
  const pill = document.getElementById("status-pill");
  pill.textContent = text;
  pill.style.color = color;
  pill.style.borderColor = color;
}

initializeApp();
