/**
 * KB Velocity Tracker (Snatch) — “on-the-fly” handedness via first backswing.
 *
 * Landmark indices used:
 * - Left: shoulder 11, wrist 15, hip 23
 * - Right: shoulder 12, wrist 16, hip 24  [web:64][web:53]
 */

import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const CONFIG = {
  // MediaPipe Pose indices
  LEFT:  { WRIST: 15, SHOULDER: 11, HIP: 23 },
  RIGHT: { WRIST: 16, SHOULDER: 12, HIP: 24 },

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

  // “First swing determines handedness”
  ARMING_COOLDOWN_MS: 700,     // prevent rapid arm flips
  DOMINANCE_RATIO: 1.15,       // motion ratio to break ties if both cross
  MOTION_EMA_ALPHA: 0.25,
  MIN_ARM_MOTION: 0.0015       // px/ms (tune per camera)
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

  // Test mode
  isTestRunning: false,
  testStage: "IDLE",          // IDLE | ARMING | RUNNING
  armedSide: "unknown",       // left | right | unknown
  lockedSide: "unknown",      // left | right | unknown
  sideLocked: false,
  lastArmedAt: 0,

  // Backswing crossing detection
  wasBelowHipLeft: false,
  wasBelowHipRight: false,

  // Motion dominance
  prevWristLeft: null,        // {xPx, yPx, tMs}
  prevWristRight: null,
  motionEmaLeft: 0,
  motionEmaRight: 0,

  // Physics (locked side only)
  prevWrist: null,            // {xNorm, yNorm, tMs}
  lockedCalibration: null,    // px/m
  smoothedVelocity: 0,

  // Rep state
  phase: "IDLE",              // IDLE -> BOTTOM -> CONCENTRIC -> LOCKOUT
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
  setStatus("Video Loaded — press Start Test (first backswing arms side)", "#fbbf24");
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
      // Autoplay blocked (expected on mobile); user will press Start Test.
      console.log("Autoplay blocked.");
    });
  }
}

/* ---------------------------- MASTER LOOP ---------------------------- */

function masterLoop() {
  if (state.isModelLoaded && state.isVideoReady && state.video.readyState >= 2) {
    // Monotonic timestamp keeps MediaPipe stable across resets/seeks
    const result = state.landmarker.detectForVideo(state.video, performance.now());

    if (result.landmarks && result.landmarks.length > 0) {
      state.lastPose = result.landmarks[0];

      // During ARMING stage, lock handedness from first backswing event
      if (state.isTestRunning && state.testStage === "ARMING") {
        tryArmFromBackswing(state.lastPose, state.video.currentTime * 1000);
      }

      // During RUNNING stage, compute snatch velocity + reps (locked side)
      if (state.isTestRunning && state.testStage === "RUNNING" && !state.video.paused) {
        runSnatchPhysicsAndLogic(state.lastPose, state.video.currentTime * 1000);
      }
    }
  }

  drawOverlay();
  requestAnimationFrame(masterLoop);
}

/* ------------------------ FIRST-SWING ARMING ------------------------ */

function tryArmFromBackswing(pose, tMs) {
  if (state.sideLocked) return;

  const L = getSideLandmarks(pose, "left");
  const R = getSideLandmarks(pose, "right");
  if (!L.wrist || !L.hip || !L.shoulder || !R.wrist || !R.hip || !R.shoulder) return;

  // Below-hip state per side
  const belowHipLeft = L.wrist.y > L.hip.y;
  const belowHipRight = R.wrist.y > R.hip.y;

  // Cross into below-hip (backswing event)
  const crossedLeft = !state.wasBelowHipLeft && belowHipLeft;
  const crossedRight = !state.wasBelowHipRight && belowHipRight;

  state.wasBelowHipLeft = belowHipLeft;
  state.wasBelowHipRight = belowHipRight;

  // Motion EMA (tie-breaker)
  const motionL = updateMotionEma("left", L.wrist, tMs);
  const motionR = updateMotionEma("right", R.wrist, tMs);

  // Cooldown: don’t re-arm rapidly
  if (state.armedSide !== "unknown" && (tMs - state.lastArmedAt) < CONFIG.ARMING_COOLDOWN_MS) return;

  let candidate = "unknown";

  // Primary: first backswing cross wins
  if (crossedLeft && !crossedRight) candidate = "left";
  if (crossedRight && !crossedLeft) candidate = "right";

  // If both cross same frame (or neither), use dominance
  if (candidate === "unknown") {
    const hasMotion = (motionL > CONFIG.MIN_ARM_MOTION) || (motionR > CONFIG.MIN_ARM_MOTION);
    if (!hasMotion) return;

    if (motionL > motionR * CONFIG.DOMINANCE_RATIO) candidate = "left";
    else if (motionR > motionL * CONFIG.DOMINANCE_RATIO) candidate = "right";
    else return;
  }

  // Candidate must actually be in below-hip zone (it’s the snatch backswing)
  if (candidate === "left" && !belowHipLeft) return;
  if (candidate === "right" && !belowHipRight) return;

  // LOCK SIDE
  state.armedSide = candidate;
  state.lockedSide = candidate;
  state.sideLocked = true;
  state.lastArmedAt = tMs;

  // IMPORTANT: first swing was only for handedness → reset all rep/physics state NOW
  resetRunStateOnly();

  // Transition to RUNNING stage; the *next* snatch(s) become the tracked reps
  state.testStage = "RUNNING";
  setStatus(`Locked ${candidate.toUpperCase()} — tracking reps now`, "#10b981");
}

function updateMotionEma(side, wrist, tMs) {
  const xPx = wrist.x * state.canvas.width;
  const yPx = wrist.y * state.canvas.height;
  const prev = side === "left" ? state.prevWristLeft : state.prevWristRight;

  let inst = 0;
  if (prev) {
    const dt = tMs - prev.tMs;
    if (dt > 0) inst = Math.hypot(xPx - prev.xPx, yPx - prev.yPx) / dt; // px/ms
  }

  const alpha = CONFIG.MOTION_EMA_ALPHA;

  if (side === "left") {
    state.motionEmaLeft = alpha * inst + (1 - alpha) * state.motionEmaLeft;
    state.prevWristLeft = { xPx, yPx, tMs };
    return state.motionEmaLeft;
  } else {
    state.motionEmaRight = alpha * inst + (1 - alpha) * state.motionEmaRight;
    state.prevWristRight = { xPx, yPx, tMs };
    return state.motionEmaRight;
  }
}

function getSideLandmarks(pose, side) {
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  return {
    wrist: pose[idx.WRIST],
    shoulder: pose[idx.SHOULDER],
    hip: pose[idx.HIP]
  };
}

/* ---------------------- SNATCH PHYSICS + LOGIC ---------------------- */

function runSnatchPhysicsAndLogic(pose, timeMs) {
  const side = state.lockedSide;
  if (side !== "left" && side !== "right") return;

  const s = getSideLandmarks(pose, side);
  const wrist = s.wrist;
  const shoulder = s.shoulder;
  const hip = s.hip;
  if (!wrist || !shoulder || !hip) return;

  // Optional: visibility guard (some models supply visibility)
  if ((wrist.visibility ?? 1) < 0.5) return;

  // Lock calibration once (px/m)
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

  // Snatch start/end zones (normalized y: smaller is higher)
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
    else if (isBelowHip) state.phase = "BOTTOM"; // failed to lock out
  }

  state.prevWrist = { xNorm: wrist.x, yNorm: wrist.y, tMs: timeMs };
}

function finishRep() {
  state.phase = "LOCKOUT";
  state.repCount++;
  state.repHistory.push(state.currentRepPeak);

  document.getElementById("val-reps").textContent = String(state.repCount);
  document.getElementById("val-peak").textContent = state.currentRepPeak.toFixed(2);

  if (state.repCount <= CONFIG.BASELINE_REPS) {
    state.baseline = avg(state.repHistory);
    const el = document.getElementById("val-drop");
    el.textContent = "CALC...";
    el.style.color = "#94a3b8";
  } else {
    const drop = (state.baseline - state.currentRepPeak) / state.baseline;
    const dropPct = (drop * 100).toFixed(1);

    const el = document.getElementById("val-drop");
    el.textContent = `-${dropPct}%`;

    if (drop * 100 >= CONFIG.DROP_FAIL) el.style.color = "#ef4444";
    else if (drop * 100 >= CONFIG.DROP_WARN) el.style.color = "#fbbf24";
    else el.style.color = "#10b981";
  }
}

/* ------------------------------ DRAWING ------------------------------ */

function drawOverlay() {
  state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
  if (!state.lastPose) return;

  // While ARMING: draw both sides (faint) + highlight the armed/locked side if set
  if (state.testStage === "ARMING" || !state.sideLocked) {
    const L = getSideLandmarks(state.lastPose, "left");
    const R = getSideLandmarks(state.lastPose, "right");

    if (L.shoulder && L.hip) drawZones(L.shoulder, L.hip, 0.20);
    if (R.shoulder && R.hip) drawZones(R.shoulder, R.hip, 0.20);

    if (L.wrist) drawDot(L.wrist, state.armedSide === "left", "#60a5fa");
    if (R.wrist) drawDot(R.wrist, state.armedSide === "right", "#f87171");
    return;
  }

  // RUNNING: draw locked side zones + dot
  const s = getSideLandmarks(state.lastPose, state.lockedSide);
  if (s.shoulder && s.hip) drawZones(s.shoulder, s.hip, 0.65);
  if (s.wrist) drawDot(s.wrist, true);
}

function drawZones(shoulder, hip, alpha = 0.6) {
  const ctx = state.ctx;
  const w = state.canvas.width;
  const h = state.canvas.height;

  // Shoulder line (lockout)
  const shoulderY = shoulder.y * h;
  ctx.strokeStyle = `rgba(16, 185, 129, ${alpha})`;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, shoulderY);
  ctx.lineTo(w, shoulderY);
  ctx.stroke();

  // Hip line (backswing bottom)
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

    // Begin the test immediately; first swing is used ONLY to arm side
    state.isTestRunning = true;
    state.testStage = "ARMING";
    state.sideLocked = false;
    state.armedSide = "unknown";
    state.lockedSide = "unknown";
    state.lastArmedAt = 0;

    // Clear run state so once side locks, we start fresh
    resetRunStateOnly();

    startBtn.textContent = "Test Running…";
    startBtn.disabled = true;
    resetBtn.disabled = false;

    setStatus("ARMING: do first backswing/snatch to pick side…", "#fbbf24");

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
  // Physics
  state.prevWrist = null;
  state.lockedCalibration = null;
  state.smoothedVelocity = 0;

  // Rep logic
  state.phase = "IDLE";
  state.repCount = 0;
  state.currentRepPeak = 0;
  state.repHistory = [];
  state.baseline = 0;

  // UI
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

  state.armedSide = "unknown";
  state.lockedSide = "unknown";
  state.sideLocked = false;
  state.lastArmedAt = 0;

  state.wasBelowHipLeft = false;
  state.wasBelowHipRight = false;

  state.prevWristLeft = null;
  state.prevWristRight = null;
  state.motionEmaLeft = 0;
  state.motionEmaRight = 0;

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
