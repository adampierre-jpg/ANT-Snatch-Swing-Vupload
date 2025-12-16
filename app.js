import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const CONFIG = {
  LEFT:  { WRIST: 15, SHOULDER: 11, HIP: 23, KNEE: 25, ANKLE: 27 },
  RIGHT: { WRIST: 16, SHOULDER: 12, HIP: 24, KNEE: 26, ANKLE: 28 },

  MIN_DET_CONF: 0.5,
  MIN_TRACK_CONF: 0.5,

  // Physics / VBT
  SMOOTHING_ALPHA: 0.30,
  MAX_REALISTIC_VELOCITY: 10.0, // m/s
  MIN_DT: 0.01,                 // s
  MAX_DT: 0.10,                 // s
  TORSO_METERS: 0.5,

  // Rep lockout criteria (overhead hold)
  LOCKOUT_VY_CUTOFF: 0.40,      // m/s (vertical near zero)
  LOCKOUT_SPEED_CUTOFF: 1.40,   // m/s (slow at lockout)
  OVERHEAD_HOLD_FRAMES: 2,      // consecutive frames

  // Drop-off
  BASELINE_REPS: 3,
  DROP_WARN: 15,
  DROP_FAIL: 20,

  // Floor “hand on bell” detection
  SHANK_FRACTION: 0.35,
  MIN_SHANK_LEN_NORM: 0.06,
  ARM_MS_REQUIRED: 100,

  // Set-down (unlock)
  RESET_MS_REQUIRED: 100,
  RESET_SPEED_CUTOFF: 0.80,
  RESET_VY_CUTOFF: 0.60,
  MIN_REPS_BEFORE_UNLOCK: 2,
  RESET_GRACE_MS_AFTER_LOCK: 500,

  // Dwell timing
  MAX_DWELL_STEP_MS: 80
};

let state = {
  video: null,
  canvas: null,
  ctx: null,
  landmarker: null,

  isModelLoaded: false,
  isVideoReady: false,
  lastPose: null,

  // Lifecycle
  isTestRunning: false,
  testStage: "IDLE", // IDLE | ARMING | RUNNING

  // Handedness
  lockedSide: "unknown",
  sideLocked: false,
  lockedAtWallMs: 0,
  lockedAtVideoMs: 0,

  // Floor dwell timers (ms, wall clock)
  floorMsLeft: 0,
  floorMsRight: 0,
  floorMsLocked: 0,
  lastWallMs: null,

  // Mechanics
  prevWrist: null,           // {xNorm, yNorm, tMs(video)}
  lockedCalibration: null,   // px/m
  smoothedVelocity: null,
  smoothedVy: null,
  lastSpeed: 0,
  lastVy: 0,

  // Totals (persist entire video)
  phase: "IDLE",
  repCount: 0,
  currentRepPeak: 0,
  repHistory: [],
  baseline: 0,
  overheadHoldCount: 0
};

/* ----------------------------- INIT ----------------------------- */

async function initializeApp() {
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

  setupControls();
  requestAnimationFrame(masterLoop);
}

function handleUpload(e) {
  const file = e.target.files?.[0];
  if (!file) return;
  resetAll("Loading…");
  state.video.srcObject = null;
  state.video.src = URL.createObjectURL(file);
  state.video.load();
}

async function startCamera() {
  resetAll("Starting camera…");
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
  state.video.srcObject = stream;
  state.video.src = "";
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
    }).catch(() => {});
  }
}

/* ---------------------------- LOOP ---------------------------- */

function masterLoop() {
  const wallMs = performance.now();

  if (state.isModelLoaded && state.isVideoReady && state.video.readyState >= 2) {
    // Monotonic ms timestamp for MediaPipe (required) [web:97]
    const result = state.landmarker.detectForVideo(state.video, wallMs);

    if (result.landmarks && result.landmarks.length > 0) {
      state.lastPose = result.landmarks[0];

      const dwellStep = getWallStepMs(wallMs);

      if (state.isTestRunning && state.testStage === "ARMING") {
        updateHandednessByFloorGrab(state.lastPose, dwellStep, wallMs);
      }
      if (state.isTestRunning && state.testStage === "RUNNING" && !state.video.paused) {
        const videoMs = state.video.currentTime * 1000;
        runSnatchPhysicsAndLogic(state.lastPose, videoMs);
        maybeResetOnSetDown(state.lastPose, dwellStep, wallMs);
      }
    } else {
      state.lastWallMs = wallMs;
    }
  } else {
    state.lastWallMs = wallMs;
  }

  drawOverlay();
  requestAnimationFrame(masterLoop);
}

function getWallStepMs(wallMs) {
  if (state.lastWallMs == null) {
    state.lastWallMs = wallMs;
    return 0;
  }
  const step = wallMs - state.lastWallMs;
  state.lastWallMs = wallMs;
  if (step < 0) return 0;
  return Math.min(step, CONFIG.MAX_DWELL_STEP_MS);
}

/* ----------------- ARMING + RESET ----------------- */

function updateHandednessByFloorGrab(pose, stepMs, wallMs) {
  if (stepMs <= 0) return;

  const inL = isWristInFloorZone(pose, "left");
  const inR = isWristInFloorZone(pose, "right");

  state.floorMsLeft = inL ? state.floorMsLeft + stepMs : 0;
  state.floorMsRight = inR ? state.floorMsRight + stepMs : 0;

  if (state.floorMsLeft >= CONFIG.ARM_MS_REQUIRED && state.floorMsRight < CONFIG.ARM_MS_REQUIRED) {
    lockSideAndStartRunning("left", wallMs);
    return;
  }
  if (state.floorMsRight >= CONFIG.ARM_MS_REQUIRED && state.floorMsLeft < CONFIG.ARM_MS_REQUIRED) {
    lockSideAndStartRunning("right", wallMs);
    return;
  }

  if (state.floorMsLeft >= CONFIG.ARM_MS_REQUIRED && state.floorMsRight >= CONFIG.ARM_MS_REQUIRED) {
    const depthL = floorDepth(pose, "left");
    const depthR = floorDepth(pose, "right");
    lockSideAndStartRunning(depthL >= depthR ? "left" : "right", wallMs);
  }
}

function lockSideAndStartRunning(side, wallMs) {
  state.lockedSide = side;
  state.sideLocked = true;
  state.lockedAtWallMs = wallMs;
  state.lockedAtVideoMs = state.video.currentTime * 1000;

  // Reset mechanics only (keep totals)
  resetSetMechanicsOnly();

  // Avoid instant set-down
  state.floorMsLocked = 0;
  state.floorMsLeft = 0;
  state.floorMsRight = 0;

  state.testStage = "RUNNING";
  setStatus(`Locked ${side.toUpperCase()} — tracking`, "#10b981");
}

function maybeResetOnSetDown(pose, stepMs, wallMs) {
  if (!state.sideLocked) return;
  if (state.repCount < CONFIG.MIN_REPS_BEFORE_UNLOCK) return;
  if ((wallMs - state.lockedAtWallMs) < CONFIG.RESET_GRACE_MS_AFTER_LOCK) return;
  if (stepMs <= 0) return;

  const inFloor = isWristInFloorZone(pose, state.lockedSide);
  const slowEnough =
    (state.lastSpeed < CONFIG.RESET_SPEED_CUTOFF) &&
    (Math.abs(state.lastVy) < CONFIG.RESET_VY_CUTOFF);

  const count = inFloor && slowEnough;
  state.floorMsLocked = count ? state.floorMsLocked + stepMs : 0;

  if (state.floorMsLocked >= CONFIG.RESET_MS_REQUIRED) {
    state.sideLocked = false;
    state.lockedSide = "unknown";
    state.lockedAtWallMs = 0;
    state.lockedAtVideoMs = 0;

    // Mechanics only (totals persist)
    resetSetMechanicsOnly();

    state.floorMsLeft = 0;
    state.floorMsRight = 0;
    state.floorMsLocked = 0;

    state.testStage = "ARMING";
    setStatus("Set down detected — grab bell to arm next set", "#fbbf24");
  }
}

function isWristInFloorZone(pose, side) {
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
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

/* ----------------- SNATCH PHYSICS + REPS ----------------- */

function runSnatchPhysicsAndLogic(pose, videoMs) {
  const side = state.lockedSide;
  if (side !== "left" && side !== "right") return;

  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const shoulder = pose[idx.SHOULDER];
  const hip = pose[idx.HIP];
  if (!wrist || !shoulder || !hip) return;

  if (!state.lockedCalibration) {
    const torsoPx = Math.abs(shoulder.y - hip.y) * state.canvas.height;
    state.lockedCalibration = torsoPx > 0 ? torsoPx / CONFIG.TORSO_METERS : 100;
  }

  if (!state.prevWrist) {
    state.prevWrist = { xNorm: wrist.x, yNorm: wrist.y, tMs: videoMs };
    return;
  }

  const dt = (videoMs - state.prevWrist.tMs) / 1000;
  if (dt < CONFIG.MIN_DT || dt > CONFIG.MAX_DT) {
    state.prevWrist = { xNorm: wrist.x, yNorm: wrist.y, tMs: videoMs };
    return;
  }

  const dxPx = (wrist.x - state.prevWrist.xNorm) * state.canvas.width;
  const dyPx = (wrist.y - state.prevWrist.yNorm) * state.canvas.height;

  const vx = (dxPx / state.lockedCalibration) / dt;
  const vy = (dyPx / state.lockedCalibration) / dt;
  const speed = Math.hypot(vx, vy);

  state.lastSpeed = speed;
  state.lastVy = vy;

  if (speed > CONFIG.MAX_REALISTIC_VELOCITY) {
    state.prevWrist = { xNorm: wrist.x, yNorm: wrist.y, tMs: videoMs };
    return;
  }

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
    if (!isBelowHip) {
      state.phase = "CONCENTRIC";
      state.currentRepPeak = 0;
      state.overheadHoldCount = 0;
    }
  } else if (state.phase === "CONCENTRIC") {
    if (v > state.currentRepPeak) state.currentRepPeak = v;

    const lockoutOk =
      isAboveShoulder &&
      Math.abs(vyS) < CONFIG.LOCKOUT_VY_CUTOFF &&
      v < CONFIG.LOCKOUT_SPEED_CUTOFF;

    if (lockoutOk) {
      state.overheadHoldCount++;
      if (state.overheadHoldCount >= CONFIG.OVERHEAD_HOLD_FRAMES) finishRep();
    } else {
      state.overheadHoldCount = 0;
      if (isBelowHip) state.phase = "BOTTOM";
    }
  }

  state.prevWrist = { xNorm: wrist.x, yNorm: wrist.y, tMs: videoMs };
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
}

/* ----------------- DRAWING ----------------- */

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

/* ----------------- CONTROLS ----------------- */

function setupControls() {
  const startBtn = document.getElementById("btn-start-test");
  const resetBtn = document.getElementById("btn-reset");

  startBtn.onclick = async () => {
    if (!state.isVideoReady) return;

    state.isTestRunning = true;
    state.testStage = "ARMING";

    state.lockedSide = "unknown";
    state.sideLocked = false;
    state.lockedAtWallMs = 0;
    state.lockedAtVideoMs = 0;

    state.floorMsLeft = 0;
    state.floorMsRight = 0;
    state.floorMsLocked = 0;
    state.lastWallMs = null;

    // Keep totals; reset per-set mechanics
    resetSetMechanicsOnly();

    startBtn.textContent = "Test Running…";
    startBtn.disabled = true;
    resetBtn.disabled = false;

    setStatus("ARMING: grab bell (wrist deep below knee) to lock side…", "#fbbf24");

    try {
      await state.video.play();
    } catch {
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

/* ----------------- RESET HELPERS ----------------- */

function resetSetMechanicsOnly() {
  state.prevWrist = null;
  state.lockedCalibration = null;
  state.smoothedVelocity = null;
  state.smoothedVy = null;
  state.lastSpeed = 0;
  state.lastVy = 0;

  state.phase = "IDLE";
  state.currentRepPeak = 0;
  state.overheadHoldCount = 0;

  document.getElementById("val-velocity").textContent = "0.00";
  document.getElementById("val-peak").textContent = "0.00";
}

function resetAll(statusText) {
  state.isTestRunning = false;
  state.testStage = "IDLE";

  state.lockedSide = "unknown";
  state.sideLocked = false;
  state.lockedAtWallMs = 0;
  state.lockedAtVideoMs = 0;

  state.floorMsLeft = 0;
  state.floorMsRight = 0;
  state.floorMsLocked = 0;
  state.lastWallMs = null;

  // Clear totals
  state.repCount = 0;
  state.repHistory = [];
  state.baseline = 0;

  resetSetMechanicsOnly();

  document.getElementById("val-reps").textContent = "0";
  const drop = document.getElementById("val-drop");
  drop.textContent = "--";
  drop.style.color = "#f1f5f9";

  const startBtn = document.getElementById("btn-start-test");
  const resetBtn = document.getElementById("btn-reset");
  startBtn.textContent = "▶ Start Test";
  startBtn.disabled = false;
  resetBtn.disabled = true;

  setStatus(statusText, "#3b82f6");
}

/* ----------------- UTILS ----------------- */

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
