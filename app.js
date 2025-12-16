/**
 * KB Velocity Tracker (Snatch) — Deployable build
 *
 * Fixes vs previous:
 * - Handedness unlock (“bell set down”) requires:
 *    (a) repCount >= 2
 *    (b) locked wrist in floor zone for >= 100ms
 *    (c) wrist speed is LOW during that dwell (prevents deep backswing false resets)
 *    (d) grace period after lock
 * - Rep counting/peak:
 *    - peak tracked during CONCENTRIC
 *    - rep ends on overhead hold (above shoulder + low speed + low vertical velocity) for N frames
 *
 * Landmark indices (MediaPipe Pose): wrists 15/16, hips 23/24, knees 25/26, ankles 27/28, shoulders 11/12. [web:53][web:54]
 */

import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const CONFIG = {
  // Pose indices
  LEFT:  { WRIST: 15, SHOULDER: 11, HIP: 23, KNEE: 25, ANKLE: 27 },
  RIGHT: { WRIST: 16, SHOULDER: 12, HIP: 24, KNEE: 26, ANKLE: 28 },

  // MediaPipe
  MIN_DET_CONF: 0.5,
  MIN_TRACK_CONF: 0.5,

  // Physics
  SMOOTHING_ALPHA: 0.30,
  MAX_REALISTIC_VELOCITY: 10.0,     // m/s
  MIN_DT: 0.01,                      // s
  MAX_DT: 0.10,                      // s
  TORSO_METERS: 0.5,                 // shoulder-to-hip approx

  // Rep logic (overhead hold)
  LOCKOUT_VY_CUTOFF: 0.40,           // m/s vertical component near zero at lockout
  LOCKOUT_SPEED_CUTOFF: 1.40,        // m/s overall speed near lockout
  OVERHEAD_HOLD_FRAMES: 2,           // must satisfy lockout criteria for N consecutive frames

  // Drop-off
  BASELINE_REPS: 3,
  DROP_WARN: 15,
  DROP_FAIL: 20,

  // Floor-zone “hand on bell”
  SHANK_FRACTION: 0.35,
  MIN_SHANK_LEN_NORM: 0.06,
  ARM_MS_REQUIRED: 100,

  // Reset (“bell set down”) — fast but safe
  RESET_MS_REQUIRED: 100,
  RESET_SPEED_CUTOFF: 0.80,          // m/s must be slow while in floor zone
  RESET_VY_CUTOFF: 0.60,             // m/s vertical speed must be modest
  MIN_REPS_BEFORE_UNLOCK: 2,
  RESET_GRACE_MS_AFTER_LOCK: 500
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
  testStage: "IDLE", // IDLE | ARMING | RUNNING

  // Handedness
  lockedSide: "unknown",
  sideLocked: false,
  lockedAtMs: 0,

  // Floor-zone dwell timers (ms)
  floorMsLeft: 0,
  floorMsRight: 0,
  floorMsLocked: 0,
  lastFloorUpdateMs: null,

  // Physics (locked side only)
  prevWrist: null,             // {xNorm, yNorm, tMs}
  lockedCalibration: null,     // px/m
  smoothedVelocity: null,
  smoothedVy: null,

  // For reset guard (latest measured values)
  lastSpeed: 0,
  lastVy: 0,

  // Rep state
  phase: "IDLE",
  repCount: 0,
  currentRepPeak: 0,
  repHistory: [],
  baseline: 0,

  // Overhead hold
  overheadHoldCount: 0
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
    const result = state.landmarker.detectForVideo(state.video, performance.now());
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
  }

  drawOverlay();
  requestAnimationFrame(masterLoop);
}

/* ----------------- FLOOR ZONE: ARMING + RESET ----------------- */

function updateHandednessByFloorGrab(pose, tMs) {
  const dtMs = updateFloorDt(tMs);
  if (dtMs <= 0) return;

  const inFloorLeft = isWristInFloorZone(pose, "left");
  const inFloorRight = isWristInFloorZone(pose, "right");

  state.floorMsLeft = inFloorLeft ? (state.floorMsLeft + dtMs) : 0;
  state.floorMsRight = inFloorRight ? (state.floorMsRight + dtMs) : 0;

  // First side to dwell >= ARM_MS_REQUIRED wins.
  if (state.floorMsLeft >= CONFIG.ARM_MS_REQUIRED && state.floorMsRight < CONFIG.ARM_MS_REQUIRED) {
    lockSideAndStartRunning("left", tMs);
    return;
  }
  if (state.floorMsRight >= CONFIG.ARM_MS_REQUIRED && state.floorMsLeft < CONFIG.ARM_MS_REQUIRED) {
    lockSideAndStartRunning("right", tMs);
    return;
  }

  // If both satisfy simultaneously, choose deeper.
  if (state.floorMsLeft >= CONFIG.ARM_MS_REQUIRED && state.floorMsRight >= CONFIG.ARM_MS_REQUIRED) {
    const depthL = floorDepth(pose, "left");
    const depthR = floorDepth(pose, "right");
    lockSideAndStartRunning(depthL >= depthR ? "left" : "right", tMs);
  }
}

function lockSideAndStartRunning(side, tMs) {
  state.lockedSide = side;
  state.sideLocked = true;
  state.lockedAtMs = tMs;

  // Clear rep/physics state so the “grab” doesn’t contaminate reps
  resetRunStateOnly();

  // Reset floor timers so the initial grab doesn’t instantly count as set-down
  state.floorMsLocked = 0;

  state.testStage = "RUNNING";
  setStatus(`Locked ${side.toUpperCase()} — tracking`, "#10b981");
  console.log(`✓ LOCKED ${side.toUpperCase()} ARM`);
}

function maybeResetOnSetDown(pose, tMs) {
  if (!state.sideLocked) return;

  // Must complete minimum reps first
  if (state.repCount < CONFIG.MIN_REPS_BEFORE_UNLOCK) return;

  // Grace after locking
  if ((tMs - state.lockedAtMs) < CONFIG.RESET_GRACE_MS_AFTER_LOCK) return;

  const dtMs = updateFloorDt(tMs);
  if (dtMs <= 0) return;

  const inFloorLocked = isWristInFloorZone(pose, state.lockedSide);

  // Critical false-reset fix:
  // Only accumulate “set down” dwell if wrist is in floor zone AND moving slowly.
  const slowEnough =
    (state.lastSpeed < CONFIG.RESET_SPEED_CUTOFF) &&
    (Math.abs(state.lastVy) < CONFIG.RESET_VY_CUTOFF);

  const countThisFrame = inFloorLocked && slowEnough;
  state.floorMsLocked = countThisFrame ? (state.floorMsLocked + dtMs) : 0;

  if (state.floorMsLocked >= CONFIG.RESET_MS_REQUIRED) {
    console.log("✓ SET DOWN — unlocking side for next set");

    state.sideLocked = false;
    state.lockedSide = "unknown";
    state.lockedAtMs = 0;

    // Reset dwell timers for re-arming
    state.floorMsLeft = 0;
    state.floorMsRight = 0;
    state.floorMsLocked = 0;
    state.lastFloorUpdateMs = tMs;

    // Prep next set
    resetRunStateOnly();
    state.testStage = "ARMING";
    setStatus("Set down detected — grab bell to arm next set", "#fbbf24");
  }
}

function updateFloorDt(tMs) {
  if (state.lastFloorUpdateMs == null) {
    state.lastFloorUpdateMs = tMs;
    return 0;
  }
  const dtMs = tMs - state.lastFloorUpdateMs;
  state.lastFloorUpdateMs = tMs;

  // Guard against seeks/jumps
  if (dtMs < 0 || dtMs > 250) return 0;
  return dtMs;
}

function isWristInFloorZone(pose, side) {
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const knee = pose[idx.KNEE];
  const ankle = pose[idx.ANKLE];
  if (!wrist || !knee || !ankle) return false;

  const shank = ankle.y - knee.y; // normalized
  if (shank < CONFIG.MIN_SHANK_LEN_NORM) return false;

  const threshold = knee.y + CONFIG.SHANK_FRACTION * shank;
  return wrist.y > threshold;
}

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

  // Lock calibration once (px/m)
  if (!state.lockedCalibration) {
    const torsoPx = Math.abs(shoulder.y - hip.y) * state.canvas.height;
    state.lockedCalibration = torsoPx > 0 ? torsoPx / CONFIG.TORSO_METERS : 100;
    console.log(`Calibration: ${state.lockedCalibration.toFixed(2)} px/m`);
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

  const vx = (dxPx / state.lockedCalibration) / dt;
  const vy = (dyPx / state.lockedCalibration) / dt; // + = down in image
  const speed = Math.hypot(vx, vy);

  // stash for reset logic
  state.lastSpeed = speed;
  state.lastVy = vy;

  if (speed > CONFIG.MAX_REALISTIC_VELOCITY) {
    state.prevWrist = { xNorm: wrist.x, yNorm: wrist.y, tMs: timeMs };
    return;
  }

  // Smooth (avoid falsy bug by using null checks)
  if (state.smoothedVelocity == null) state.smoothedVelocity = speed;
  if (state.smoothedVy == null) state.smoothedVy = vy;

  state.smoothedVelocity =
    CONFIG.SMOOTHING_ALPHA * speed + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVelocity;

  state.smoothedVy =
    CONFIG.SMOOTHING_ALPHA * vy + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVy;

  const v = state.smoothedVelocity;
  const vyS = state.smoothedVy;

  document.getElementById("val-velocity").textContent = v.toFixed(2);

  const isBelowHip = wrist.y > hip.y;
  const isAboveShoulder = wrist.y < shoulder.y;

  if (state.phase === "IDLE" || state.phase === "LOCKOUT") {
    if (isBelowHip) {
      state.phase = "BOTTOM";
      state.overheadHoldCount = 0;
    }
  } else if (state.phase === "BOTTOM") {
    // Concentric starts once wrist rises above hip line
    if (!isBelowHip) {
      state.phase = "CONCENTRIC";
      state.currentRepPeak = 0;
      state.overheadHoldCount = 0;
    }
  } else if (state.phase === "CONCENTRIC") {
    // Peak speed per rep
    if (v > state.currentRepPeak) state.currentRepPeak = v;

    // Lockout hold
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
      if (isBelowHip) state.phase = "BOTTOM";
    }
  }

  state.prevWrist = { xNorm: wrist.x, yNorm: wrist.y, tMs: timeMs };
}

function finishRep() {
  state.phase = "LOCKOUT";
  state.overheadHoldCount = 0;

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

  console.log(`✓ REP ${state.repCount} | Peak ${state.currentRepPeak.toFixed(2)} m/s`);
}

/* ------------------------------ DRAWING ------------------------------ */

function drawOverlay() {
  state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
  if (!state.lastPose) return;

  if (state.isTestRunning && state.testStage === "ARMING") {
    drawFloorDebug(state.lastPose, "left");
    drawFloorDebug(state.lastPose, "right");

    const Lw = state.lastPose[CONFIG.LEFT.WRIST];
    const Rw = state.lastPose[CONFIG.RIGHT.WRIST];
    if (Lw) drawDot(Lw, false, "#60a5fa");
    if (Rw) drawDot(Rw, false, "#f87171");
    return;
  }

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

  state.ctx.strokeStyle = side === "left" ? "rgba(96,165,250,0.4)" : "rgba(248,113,113,0.4)";
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

    state.lockedSide = "unknown";
    state.sideLocked = false;
    state.lockedAtMs = 0;

    state.floorMsLeft = 0;
    state.floorMsRight = 0;
    state.floorMsLocked = 0;
    state.lastFloorUpdateMs = null;

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
  state.smoothedVelocity = null;
  state.smoothedVy = null;

  state.lastSpeed = 0;
  state.lastVy = 0;

  state.phase = "IDLE";
  state.repCount = 0;
  state.currentRepPeak = 0;
  state.repHistory = [];
  state.baseline = 0;
  state.overheadHoldCount = 0;

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
  state.lockedAtMs = 0;

  state.floorMsLeft = 0;
  state.floorMsRight = 0;
  state.floorMsLocked = 0;
  state.lastFloorUpdateMs = null;

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
