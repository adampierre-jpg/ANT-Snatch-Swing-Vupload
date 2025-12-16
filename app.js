import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

/**
 * Auto-handedness lock for snatch:
 * - A side becomes "armed" when its wrist crosses BELOW its hip line (backswing event).
 * - Once armed, we still require some motion dominance for a few frames to avoid false arming.
 * - When Start Test is pressed, the armed side becomes locked for the test.
 */

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

  // Handedness lock logic
  ARMING_COOLDOWN_MS: 600,     // once armed, ignore other side for this long
  DOMINANCE_RATIO: 1.15,       // require 15% higher motion to confirm
  MOTION_EMA_ALPHA: 0.25,
  MIN_ARM_MOTION: 0.0015       // px/ms baseline activity threshold (tune)
};

let state = {
  video: null,
  canvas: null,
  ctx: null,
  landmarker: null,

  isModelLoaded: false,
  isVideoReady: false,
  isTestRunning: false,

  // Pose cache
  lastPose: null,

  // Active side selection
  armedSide: "unknown",      // "left" | "right" | "unknown"
  lockedSide: "unknown",     // locked when test starts
  sideLocked: false,
  lastArmedAt: 0,

  // For detecting backswing crossing
  wasBelowHipLeft: false,
  wasBelowHipRight: false,

  // For motion dominance
  prevWristLeft: null,       // {xPx, yPx, tMs}
  prevWristRight: null,
  motionEmaLeft: 0,
  motionEmaRight: 0,

  // Physics (active side only)
  prevWrist: null,           // {xNorm, yNorm, tMs} for active wrist
  lockedCalibration: null,   // px/m
  smoothedVelocity: 0,

  // Rep state
  phase: "IDLE",
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

    console.log("🚀 Initializing MediaPipe PoseLandmarker...");

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

  fullResetAll("Loading…");
  state.isVideoReady = false;
  state.lastPose = null;

  state.video.srcObject = null;
  state.video.src = URL.createObjectURL(file);
  state.video.load();
}

async function startCamera() {
  try {
    fullResetAll("Starting camera…");
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

  setStatus("Video Loaded — do a backswing to arm side…", "#fbbf24");
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
      console.log("Autoplay blocked (expected on some mobile browsers).");
    });
  }
}

/* ---------------------------- MASTER LOOP ---------------------------- */

function masterLoop() {
  if (state.isModelLoaded && state.isVideoReady && state.video.readyState >= 2) {
    const result = state.landmarker.detectForVideo(state.video, performance.now());

    if (result.landmarks && result.landmarks.length > 0) {
      state.lastPose = result.landmarks[0];

      // Arm the side using backswing-cross + dominance
      updateArmedSide(state.lastPose, state.video.currentTime * 1000);

      if (state.isTestRunning && !state.video.paused) {
        runSnatchPhysicsAndLogic(state.lastPose, state.video.currentTime * 1000);
      }
    }
  }

  drawOverlay();
  requestAnimationFrame(masterLoop);
}

/* ------------------------ HANDEDNESS LOCK ------------------------ */

function updateArmedSide(pose, tMs) {
  if (state.sideLocked) return;

  const L = getSideLandmarks(pose, "left");
  const R = getSideLandmarks(pose, "right");

  if (!L.wrist || !L.hip || !L.shoulder || !R.wrist || !R.hip || !R.shoulder) {
    // not enough info to arm
    return;
  }

  // Determine "below hip" per side (backswing zone)
  const belowHipLeft = L.wrist.y > L.hip.y;
  const belowHipRight = R.wrist.y > R.hip.y;

  // Detect crossing into below-hip zone (from not-below -> below)
  const crossedLeft = !state.wasBelowHipLeft && belowHipLeft;
  const crossedRight = !state.wasBelowHipRight && belowHipRight;

  state.wasBelowHipLeft = belowHipLeft;
  state.wasBelowHipRight = belowHipRight;

  // Update motion EMAs (px/ms) for dominance checks
  const motionL = updateMotionEma("left", L.wrist, tMs);
  const motionR = updateMotionEma("right", R.wrist, tMs);

  // If already armed recently, ignore rapid flips
  if (state.armedSide !== "unknown" && (tMs - state.lastArmedAt) < CONFIG.ARMING_COOLDOWN_MS) {
    return;
  }

  // Rule 1: if only one side crosses, prefer that side
  let candidate = "unknown";
  if (crossedLeft && !crossedRight) candidate = "left";
  if (crossedRight && !crossedLeft) candidate = "right";

  // Rule 2: if both crossed (or neither), use motion dominance
  if (candidate === "unknown") {
    // Must have some baseline motion first
    const hasMotion = (motionL > CONFIG.MIN_ARM_MOTION) || (motionR > CONFIG.MIN_ARM_MOTION);
    if (!hasMotion) return;

    if (motionL > motionR * CONFIG.DOMINANCE_RATIO) candidate = "left";
    else if (motionR > motionL * CONFIG.DOMINANCE_RATIO) candidate = "right";
    else return; // ambiguous
  }

  // Confirm candidate only if it is actually in backswing zone (below hip)
  if (candidate === "left" && !belowHipLeft) return;
  if (candidate === "right" && !belowHipRight) return;

  state.armedSide = candidate;
  state.lastArmedAt = tMs;

  setStatus(`Armed: ${candidate.toUpperCase()} — press Start Test`, "#10b981");
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
  const wrist = pose[idx.WRIST];
  const shoulder = pose[idx.SHOULDER];
  const hip = pose[idx.HIP];
  return { wrist, shoulder, hip };
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

  if (state.repCount <= CONFIG.BASELINE_REPS) {
    state.baseline = avg(state.repHistory);
    document.getElementById("val-drop").textContent = "CALC...";
    document.getElementById("val-drop").style.color = "#94a3b8";
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

  // Draw zones/dot for armed side if available; otherwise draw both faintly.
  if (state.armedSide === "left" || state.armedSide === "right") {
    const s = getSideLandmarks(state.lastPose, state.armedSide);
    if (s.shoulder && s.hip) drawZones(s.shoulder, s.hip, 0.65);
    if (s.wrist) drawDot(s.wrist, state.armedSide === state.lockedSide);
  } else {
    const L = getSideLandmarks(state.lastPose, "left");
    const R = getSideLandmarks(state.lastPose, "right");
    if (L.shoulder && L.hip) drawZones(L.shoulder, L.hip, 0.20);
    if (R.shoulder && R.hip) drawZones(R.shoulder, R.hip, 0.20);
    if (L.wrist) drawDot(L.wrist, false, "#60a5fa");
    if (R.wrist) drawDot(R.wrist, false, "#f87171");
  }
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

function drawDot(wrist, locked = false, overrideColor = null) {
  const x = wrist.x * state.canvas.width;
  const y = wrist.y * state.canvas.height;

  const ctx = state.ctx;
  ctx.fillStyle = overrideColor || (locked ? "#10b981" : "#ef4444");
  ctx.beginPath();
  ctx.arc(x, y, locked ? 10 : 7, 0, 2 * Math.PI);
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
    if (state.armedSide !== "left" && state.armedSide !== "right") {
      alert("Do a backswing first so the app can arm LEFT or RIGHT.");
      return;
    }

    // Lock side
    state.sideLocked = true;
    state.lockedSide = state.armedSide;

    // Reset test state
    state.isTestRunning = true;
    state.phase = "IDLE";
    state.repCount = 0;
    state.repHistory = [];
    state.baseline = 0;
    state.currentRepPeak = 0;

    state.prevWrist = null;
    state.lockedCalibration = null;
    state.smoothedVelocity = 0;

    document.getElementById("val-velocity").textContent = "0.00";
    document.getElementById("val-peak").textContent = "0.00";
    document.getElementById("val-reps").textContent = "0";
    document.getElementById("val-drop").textContent = "--";
    document.getElementById("val-drop").style.color = "#f1f5f9";

    startBtn.textContent = "Test Running…";
    startBtn.disabled = true;
    resetBtn.disabled = false;

    setStatus(`TEST RUNNING — ${state.lockedSide.toUpperCase()} ARM`, "#10b981");

    try {
      await state.video.play();
    } catch {
      alert("Playback blocked. Tap video area once, then press Start Test again.");
      startBtn.textContent = "▶ Start Test";
      startBtn.disabled = false;
      resetBtn.disabled = true;
      state.isTestRunning = false;
      state.sideLocked = false;
      state.lockedSide = "unknown";
    }
  };

  resetBtn.onclick = () => {
    state.video.pause();
    state.video.currentTime = 0;

    fullResetAll("Reset — do a backswing to arm side…");
    primeVideo();
  };
}

/* ------------------------------ RESET ------------------------------ */

function fullResetAll(statusText) {
  state.isTestRunning = false;

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
  document.getElementById("val-drop").textContent = "--";
  document.getElementById("val-drop").style.color = "#f1f5f9";

  const startBtn = document.getElementById("btn-start-test");
  const resetBtn = document.getElementById("btn-reset");
  startBtn.textContent = "▶ Start Test";
  startBtn.disabled = false;
  resetBtn.disabled = true;

  setStatus(statusText, "#fbbf24");
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
