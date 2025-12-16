/**
 * KB Velocity Tracker (Snatch)
 * Handedness:
 *  - When you press Start Test, app enters ARMING.
 *  - The first wrist that goes clearly below its same-side knee (scaled by shank length)
 *    for ~2–3 frames is treated as the "grab" hand and becomes the locked side.
 *  - When that same wrist returns to that floor zone and dwells briefly (~100ms) with low
 *    velocity, it is treated as bell set-down and handedness is reset for the next set.
 *
 * Rep logic, smoothing, calibration, and UI wiring are preserved from the previous version.
 */

import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const CONFIG = {
  // Pose indices (MediaPipe Pose)
  LEFT:  { WRIST: 15, SHOULDER: 11, HIP: 23, KNEE: 25, ANKLE: 27 }, // [web:54][web:53]
  RIGHT: { WRIST: 16, SHOULDER: 12, HIP: 24, KNEE: 26, ANKLE: 28 }, // [web:54][web:53]

  // MediaPipe thresholds
  MIN_DET_CONF: 0.5,
  MIN_TRACK_CONF: 0.5,

  // Velocity stabilizers
  SMOOTHING_ALPHA: 0.30,
  MAX_REALISTIC_VELOCITY: 10.0,
  MIN_DT: 0.005,
  MAX_DT: 0.12,
  TORSO_METERS: 0.5,

  // Snatch rep logic
  LOCKOUT_VEL_CUTOFF: 0.5,
  BASELINE_REPS: 3,
  DROP_WARN: 15,
  DROP_FAIL: 20,

  // Floor-zone handedness detection (hike pass / hand on bell)
  SHANK_FRACTION: 0.30,       // wrist must be this fraction of shank below knee
  MIN_SHANK_LEN_NORM: 0.03,   // ignore bad pose if knee/ankle too close
  ARM_FRAMES_REQUIRED: 2,     // ~1–2 frames to arm (fast grab)
  RESET_FRAMES_REQUIRED: 3,   // ~100ms at ~30fps for set-down
  RESET_VEL_THRESHOLD: 0.35   // must be slow while in floor zone to reset
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

  // Test stage
  isTestRunning: false,
  testStage: "IDLE",            // IDLE | ARMING | RUNNING

  // Handedness
  lockedSide: "unknown",        // "left" | "right"
  sideLocked: false,

  // Floor-zone dwell counts
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
    const result = state.landmarker.detectForVideo(state.video, performance.now());

    if (result.landmarks && result.landmarks.length > 0) {
      state.lastPose = result.landmarks[0];

      if (state.isTestRunning && state.testStage === "ARMING") {
        updateHandednessByFloorGrab(state.lastPose);
      }

      if (state.isTestRunning && state.testStage === "RUNNING" && !state.video.paused) {
        runSnatchPhysicsAndLogic(state.lastPose, state.video.currentTime * 1000);
        maybeResetOnSetDown(state.lastPose);
      }
    }
  }

  drawOverlay();
  requestAnimationFrame(masterLoop);
}

/* ----------------- FLOOR-ZONE HANDENESS + RESET ----------------- */

function updateHandednessByFloorGrab(pose) {
  const inFloorLeft = isWristInFloorZone(pose, "left");
  const inFloorRight = isWristInFloorZone(pose, "right");

  state.floorCountLeft = inFloorLeft ? state.floorCountLeft + 1 : 0;
  state.floorCountRight = inFloorRight ? state.floorCountRight + 1 : 0;

  let chosen = "unknown";

  if (state.floorCountLeft >= CONFIG.ARM_FRAMES_REQUIRED &&
      state.floorCountRight < CONFIG.ARM_FRAMES_REQUIRED) {
    chosen = "left";
  } else if (state.floorCountRight >= CONFIG.ARM_FRAMES_REQUIRED &&
             state.floorCountLeft < CONFIG.ARM_FRAMES_REQUIRED) {
    chosen = "right";
  } else if (state.floorCountLeft >= CONFIG.ARM_FRAMES_REQUIRED &&
             state.floorCountRight >= CONFIG.ARM_FRAMES_REQUIRED) {
    // both hit at once: choose deeper one
    const depthL = floorDepth(pose, "left");
    const depthR = floorDepth(pose, "right");
    chosen = depthL >= depthR ? "left" : "right";
  }

  if (chosen !== "unknown") {
    lockSideAndStartRunning(chosen);
  }
}

function lockSideAndStartRunning(side) {
  state.lockedSide = side;
  state.sideLocked = true;

  resetRunStateOnly();

  state.testStage = "RUNNING";
  setStatus(`Locked ${side.toUpperCase()} — tracking`, "#10b981");
}

function maybeResetOnSetDown(pose) {
  if (!state.sideLocked) return;

  const inFloor = isWristInFloorZone(pose, state.lockedSide);
  const lowVel = (state.smoothedVelocity || 0) < CONFIG.RESET_VEL_THRESHOLD;

  if (inFloor && lowVel) {
    state.floorCountLocked += 1;
  } else {
    state.floorCountLocked = 0;
  }

  if (state.floorCountLocked >= CONFIG.RESET_FRAMES_REQUIRED) {
    // auto reset handedness for next set, keep video rolling
    state.sideLocked = false;
    state.lockedSide = "unknown";
    state.testStage = "ARMING";

    state.floorCountLeft = 0;
    state.floorCountRight = 0;
    state.floorCountLocked = 0;

    resetRunStateOnly();
    setStatus("Bell set down — grab again to arm next side", "#fbbf24");
  }
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

    if (isAboveShoulder && velocity < CONFIG.LOCKOUT_VEL_CUTOFF) {
      finishRep();
    } else if (isBelowHip) {
      state.phase = "BOTTOM";
    }
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

  if (state.isTestRunning && state.testStage === "ARMING") {
    // Show floor thresholds + wrists for both sides
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

  state.ctx.strokeStyle = side === "left"
    ? "rgba(96,165,250,0.35)"
    : "rgba(248,113,113,0.35)";
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
    state.floorCountLeft = 0;
    state.floorCountRight = 0;
    state.floorCountLocked = 0;

    resetRunStateOnly();

    startBtn.textContent = "Test Running…";
    startBtn.disabled = true;
    resetBtn.disabled = false;

    setStatus("ARMING: grab bell (wrist clearly below knee) to lock side…", "#fbbf24");

    try {
      await state.video.play();
    } catch {
      alert("Playback blocked. Tap the video once, then press Start Test again.");
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
