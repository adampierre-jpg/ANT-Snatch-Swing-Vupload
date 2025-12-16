/**
 * KB Velocity Tracker (Snatch) — Floor “hand-on-bell” handedness + reliable reps/peaks
 *
 * Key fixes:
 * 1. Handedness UNLOCK (bell set down) only after ≥2 reps completed.
 * 2. Arming/reset dwell is time-based (~100ms).
 * 3. Robust lockout detection using vertical velocity + brief overhead hold.
 *
 * Pose landmark indices: MediaPipe Pose (33 landmarks)
 */

import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const CONFIG = {
  // Pose indices
  LEFT:  { WRIST: 15, SHOULDER: 11, HIP: 23, KNEE: 25, ANKLE: 27 },
  RIGHT: { WRIST: 16, SHOULDER: 12, HIP: 24, KNEE: 26, ANKLE: 28 },

  // MediaPipe
  MIN_DET_CONF: 0.5,
  MIN_TRACK_CONF: 0.5,

  // Velocity calculation & smoothing
  SMOOTHING_ALPHA: 0.30,
  MAX_REALISTIC_VELOCITY: 10.0,
  MIN_DT: 0.01,
  MAX_DT: 0.10,
  TORSO_METERS: 0.5,

  // Lockout detection
  LOCKOUT_VY_CUTOFF: 0.35,        // m/s (stricter = longer pause required)
  LOCKOUT_SPEED_CUTOFF: 1.25,     // m/s overall (prevents counting mid-flight)
  OVERHEAD_HOLD_FRAMES: 2,        // consecutive frames meeting criteria

  // Rep analysis
  BASELINE_REPS: 3,
  DROP_WARN: 15,   // %
  DROP_FAIL: 20,   // %

  // Floor-zone handedness
  SHANK_FRACTION: 0.35,
  MIN_SHANK_LEN_NORM: 0.06,

  // Dwell timings (ms)
  ARM_MS_REQUIRED: 100,
  RESET_MS_REQUIRED: 100,
  RESET_GRACE_MS_AFTER_LOCK: 500,

  // Safety
  MIN_REPS_BEFORE_UNLOCK: 2,
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

  // Test state
  isTestRunning: false,
  testStage: "IDLE", // IDLE | ARMING | RUNNING

  // Handedness
  lockedSide: "unknown", // "left" | "right"
  sideLocked: false,
  lockedAtMs: 0,

  // Floor dwell timers (ms)
  floorDwell: { left: 0, right: 0, locked: 0 },
  lastFloorTs: null,

  // Physics (locked side only)
  prevWrist: null,            // {xNorm, yNorm, ts}
  calibration: null,          // px/m
  smoothedVelocity: 0,
  smoothedVy: 0,

  // Rep state
  phase: "IDLE",              // IDLE → BOTTOM → CONCENTRIC → LOCKOUT
  repCount: 0,
  currentRepPeak: 0,
  repHistory: [],
  baseline: 0,
  overheadHoldCount: 0,
};

/* ----------------------------- INIT ----------------------------- */

async function initializeApp() {
  try {
    state.video = document.getElementById("video");
    state.canvas = document.getElementById("canvas");
    state.ctx = state.canvas.getContext("2d");

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );

    state.landmarker = await PoseLandmarker.createFromOptions(vision, {
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
      alert("Video error: " + (state.video?.error?.message || "Unknown"));
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
  setStatus("Video ready — press Start Test", "#fbbf24");
}

function primeVideo() {
  const playPromise = state.video.play();
  if (playPromise) {
    playPromise.then(() => {
      setTimeout(() => {
        if (!state.isTestRunning) state.video.pause();
        state.video.currentTime = 0;
      }, 120);
    }).catch(() => console.log("Autoplay blocked"));
  }
}

/* ---------------------------- MASTER LOOP ---------------------------- */

function masterLoop() {
  if (state.isModelLoaded && state.isVideoReady && state.video.readyState >= 2) {
    const timestamp = performance.now();
    const result = state.landmarker.detectForVideo(state.video, timestamp);

    if (result.landmarks?.length > 0) {
      state.lastPose = result.landmarks[0];
      const tMs = timestamp;

      if (state.isTestRunning) {
        if (state.testStage === "ARMING") {
          updateHandednessArming(state.lastPose, tMs);
        } else if (state.testStage === "RUNNING" && !state.video.paused) {
          updateSnatchPhysics(state.lastPose, tMs);
          checkSetDownReset(state.lastPose, tMs);
        }
      }
    }
  }

  drawOverlay();
  requestAnimationFrame(masterLoop);
}

/* ----------------- FLOOR ZONE: ARMING & RESET ----------------- */

function updateDwellTimer(currentTs) {
  if (state.lastFloorTs === null) {
    state.lastFloorTs = currentTs;
    return 0;
  }
  let dt = currentTs - state.lastFloorTs;
  state.lastFloorTs = currentTs;
  if (dt < 0 || dt > 250) return 0; // guard jumps/seeks
  return dt;
}

function isWristInFloorZone(pose, side) {
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const { wrist, knee, ankle } = { wrist: pose[idx.WRIST], knee: pose[idx.KNEE], ankle: pose[idx.ANKLE] };
  if (!wrist || !knee || !ankle) return false;

  const shank = ankle.y - knee.y;
  if (shank < CONFIG.MIN_SHANK_LEN_NORM) return false;

  const threshold = knee.y + CONFIG.SHANK_FRACTION * shank;
  return wrist.y > threshold;
}

function floorDepth(pose, side) {
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const { wrist, knee, ankle } = { wrist: pose[idx.WRIST], knee: pose[idx.KNEE], ankle: pose[idx.ANKLE] };
  if (!wrist || !knee || !ankle) return 0;

  const shank = ankle.y - knee.y;
  if (shank < CONFIG.MIN_SHANK_LEN_NORM) return 0;

  const threshold = knee.y + CONFIG.SHANK_FRACTION * shank;
  return Math.max(0, wrist.y - threshold);
}

function updateHandednessArming(pose, tMs) {
  const dt = updateDwellTimer(tMs);
  if (dt <= 0) return;

  const inLeft = isWristInFloorZone(pose, "left");
  const inRight = isWristInFloorZone(pose, "right");

  state.floorDwell.left  = inLeft  ? state.floorDwell.left  + dt : 0;
  state.floorDwell.right = inRight ? state.floorDwell.right + dt : 0;

  // First to reach threshold wins
  if (state.floorDwell.left >= CONFIG.ARM_MS_REQUIRED && state.floorDwell.right < CONFIG.ARM_MS_REQUIRED) {
    lockSide("left", tMs);
  } else if (state.floorDwell.right >= CONFIG.ARM_MS_REQUIRED && state.floorDwell.left < CONFIG.ARM_MS_REQUIRED) {
    lockSide("right", tMs);
  } else if (state.floorDwell.left >= CONFIG.ARM_MS_REQUIRED && state.floorDwell.right >= CONFIG.ARM_MS_REQUIRED) {
    // Tie → choose deeper wrist
    const depthL = floorDepth(pose, "left");
    const depthR = floorDepth(pose, "right");
    lockSide(depthL >= depthR ? "left" : "right", tMs);
  }
}

function checkSetDownReset(pose, tMs) {
  if (!state.sideLocked) return;
  if (state.repCount < CONFIG.MIN_REPS_BEFORE_UNLOCK) return;
  if ((tMs - state.lockedAtMs) < CONFIG.RESET_GRACE_MS_AFTER_LOCK) return;

  const dt = updateDwellTimer(tMs);
  if (dt <= 0) return;

  const inFloor = isWristInFloorZone(pose, state.lockedSide);
  state.floorDwell.locked = inFloor ? state.floorDwell.locked + dt : 0;

  if (state.floorDwell.locked >= CONFIG.RESET_MS_REQUIRED) {
    console.log("✓ BELL SET DOWN — unlocking for next set");
    state.sideLocked = false;
    state.lockedSide = "unknown";
    state.lockedAtMs = 0;
    Object.assign(state.floorDwell, { left: 0, right: 0, locked: 0 });
    state.lastFloorTs = null;

    resetRunStateOnly();
    state.testStage = "ARMING";
    setStatus("Set down detected — grab bell to arm next set", "#fbbf24");
  }
}

function lockSide(side, tMs) {
  state.lockedSide = side;
  state.sideLocked = true;
  state.lockedAtMs = tMs;
  state.floorDwell.locked = 0;

  resetRunStateOnly();
  state.testStage = "RUNNING";
  setStatus(`Locked ${side.toUpperCase()} — tracking`, "#10b981");
  console.log(`✓ LOCKED ${side.toUpperCase()} HAND`);
}

/* ---------------------- SNATCH PHYSICS & REP LOGIC ---------------------- */

function updateSnatchPhysics(pose, timestamp) {
  const side = state.lockedSide;
  if (side !== "left" && side !== "right") return;

  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const shoulder = pose[idx.SHOULDER];
  const hip = pose[idx.HIP];

  if (!wrist || !shoulder || !hip || (wrist.visibility ?? 1) < 0.5) return;

  // Calibrate scale once
  if (!state.calibration) {
    const torsoPx = Math.abs(shoulder.y - hip.y) * state.canvas.height;
    state.calibration = torsoPx > 0 ? torsoPx / CONFIG.TORSO_METERS : 100;
    console.log(`Calibration: ${state.calibration.toFixed(2)} px/m`);
  }

  // First frame — initialise
  if (!state.prevWrist) {
    state.prevWrist = { xNorm: wrist.x, yNorm: wrist.y, ts: timestamp };
    return;
  }

  const dt = (timestamp - state.prevWrist.ts) / 1000;
  if (dt < CONFIG.MIN_DT || dt > CONFIG.MAX_DT) {
    state.prevWrist = { xNorm: wrist.x, yNorm: wrist.y, ts: timestamp };
    return;
  }

  // Raw pixel delta → m/s
  const dxPx = (wrist.x - state.prevWrist.xNorm) * state.canvas.width;
  const dyPx = (wrist.y - state.prevWrist.yNorm) * state.canvas.height;
  const vx = dxPx / state.calibration / dt;
  const vy = dyPx / state.calibration / dt; // +ve = down
  const speed = Math.hypot(vx, vy);

  if (speed > CONFIG.MAX_REALISTIC_VELOCITY) {
    state.prevWrist = { xNorm: wrist.x, yNorm: wrist.y, ts: timestamp };
    return;
  }

  // Exponential smoothing
  state.smoothedVelocity = CONFIG.SMOOTHING_ALPHA * speed +
                          (1 - CONFIG.SMOOTHING_ALPHA) * (state.smoothedVelocity || speed);
  state.smoothedVy = CONFIG.SMOOTHING_ALPHA * vy +
                     (1 - CONFIG.SMOOTHING_ALPHA) * (state.smoothedVy || vy);

  const v = state.smoothedVelocity;
  const vyS = state.smoothedVy;
  document.getElementById("val-velocity").textContent = v.toFixed(2);

  // Zones
  const belowHip = wrist.y > hip.y;
  const aboveShoulder = wrist.y < shoulder.y;

  // State machine
  switch (state.phase) {
    case "IDLE":
    case "LOCKOUT":
      if (belowHip) {
        state.phase = "BOTTOM";
        state.overheadHoldCount = 0;
      }
      break;

    case "BOTTOM":
      if (!belowHip) {
        state.phase = "CONCENTRIC";
        state.currentRepPeak = 0;
        state.overheadHoldCount = 0;
      }
      break;

    case "CONCENTRIC":
      if (v > state.currentRepPeak) state.currentRepPeak = v;

      // Lockout criteria
      if (aboveShoulder && Math.abs(vyS) < CONFIG.LOCKOUT_VY_CUTOFF && v < CONFIG.LOCKOUT_SPEED_CUTOFF) {
        state.overheadHoldCount++;
        if (state.overheadHoldCount >= CONFIG.OVERHEAD_HOLD_FRAMES) {
          completeRep();
        }
      } else {
        state.overheadHoldCount = 0;
        if (belowHip) state.phase = "BOTTOM";
      }
      break;
  }

  state.prevWrist = { xNorm: wrist.x, yNorm: wrist.y, ts: timestamp };
}

function completeRep() {
  state.phase = "LOCKOUT";
  state.overheadHoldCount = 0;
  state.repCount++;
  state.repHistory.push(state.currentRepPeak);

  document.getElementById("val-reps").textContent = state.repCount;
  document.getElementById("val-peak").textContent = state.currentRepPeak.toFixed(2);

  const dropEl = document.getElementById("val-drop");

  if (state.repCount <= CONFIG.BASELINE_REPS) {
    state.baseline = avg(state.repHistory);
    dropEl.textContent = "CALC...";
    dropEl.style.color = "#94a3b8";
  } else {
    const dropPct = ((state.baseline - state.currentRepPeak) / state.baseline * 100).toFixed(1);
    dropEl.textContent = `-${dropPct}%`;
    if (dropPct >= CONFIG.DROP_FAIL) dropEl.style.color = "#ef4444";
    else if (dropPct >= CONFIG.DROP_WARN) dropEl.style.color = "#fbbf24";
    else dropEl.style.color = "#10b981";
  }

  console.log(`✓ REP ${state.repCount} | Peak ${state.currentRepPeak.toFixed(2)} m/s`);
}

/* ------------------------------ DRAWING ------------------------------ */

function drawOverlay() {
  state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
  if (!state.lastPose) return;

  if (state.testStage === "ARMING") {
    drawFloorDebug(state.lastPose, "left");
    drawFloorDebug(state.lastPose, "right");
    if (state.lastPose[CONFIG.LEFT.WRIST])  drawDot(state.lastPose[CONFIG.LEFT.WRIST],  false, "#60a5fa");
    if (state.lastPose[CONFIG.RIGHT.WRIST]) drawDot(state.lastPose[CONFIG.RIGHT.WRIST], false, "#f87171");
    return;
  }

  if (state.sideLocked) {
    const idx = state.lockedSide === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
    const wrist = state.lastPose[idx.WRIST];
    const shoulder = state.lastPose[idx.SHOULDER];
    const hip = state.lastPose[idx.HIP];

    if (shoulder && hip) drawZones(shoulder, hip);
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
  state.ctx.setLineDash([6, 4]);
  state.ctx.beginPath();
  state.ctx.moveTo(0, yPx);
  state.ctx.lineTo(state.canvas.width, yPx);
  state.ctx.stroke();
  state.ctx.setLineDash([]);
}

function drawZones(shoulder, hip) {
  const { ctx, canvas } = state;
  const w = canvas.width;
  const h = canvas.height;

  // Shoulder line (overhead)
  ctx.strokeStyle = "rgba(16, 185, 129, 0.65)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, shoulder.y * h);
  ctx.lineTo(w, shoulder.y * h);
  ctx.stroke();

  // Hip line (bottom)
  ctx.strokeStyle = "rgba(59, 130, 246, 0.65)";
  ctx.beginPath();
  ctx.moveTo(0, hip.y * h);
  ctx.lineTo(w, hip.y * h);
  ctx.stroke();
}

function drawDot(wrist, emphasized = false, color = null) {
  const x = wrist.x * state.canvas.width;
  const y = wrist.y * state.canvas.height;
  const ctx = state.ctx;

  ctx.fillStyle = color || (emphasized ? "#10b981" : "#ef4444");
  ctx.beginPath();
  ctx.arc(x, y, emphasized ? 10 : 7, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "white";
  ctx.beginPath();
  ctx.arc(x, y, 3, 0, Math.PI * 2);
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

    // Reset handedness & timers
    state.lockedSide = "unknown";
    state.sideLocked = false;
    state.lockedAtMs = 0;
    Object.assign(state.floorDwell, { left: 0, right: 0, locked: 0 });
    state.lastFloorTs = null;

    resetRunStateOnly();

    startBtn.textContent = "Test Running…";
    startBtn.disabled = true;
    resetBtn.disabled = false;

    setStatus("ARMING: place hand on bell to lock side…", "#fbbf24");

    try {
      await state.video.play();
    } catch {
      alert("Playback blocked — tap video then try again.");
      resetAll("Ready");
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
  state.calibration = null;
  state.smoothedVelocity = 0;
  state.smoothedVy = 0;

  state.phase = "IDLE";
  state.repCount = 0;
  state.currentRepPeak = 0;
  state.repHistory = [];
  state.baseline = 0;
  state.overheadHoldCount = 0;

  document.getElementById("val-velocity").textContent = "0.00";
  document.getElementById("val-peak").textContent = "0.00";
  document.getElementById("val-reps").textContent = "0";
  const dropEl = document.getElementById("val-drop");
  dropEl.textContent = "--";
  dropEl.style.color = "#f1f5f9";
}

function resetAll(statusText) {
  state.isTestRunning = false;
  state.testStage = "IDLE";

  state.lockedSide = "unknown";
  state.sideLocked = false;
  state.lockedAtMs = 0;
  Object.assign(state.floorDwell, { left: 0, right: 0, locked: 0 });
  state.lastFloorTs = null;

  resetRunStateOnly();

  document.getElementById("btn-start-test").textContent = "▶ Start Test";
  document.getElementById("btn-start-test").disabled = false;
  document.getElementById("btn-reset").disabled = true;

  setStatus(statusText, "#3b82f6");
}

/* ------------------------------ UTILS ------------------------------ */

function avg(arr) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function setStatus(text, color) {
  const el = document.getElementById("status-pill");
  el.textContent = text;
  el.style.color = color;
  el.style.borderColor = color;
}

initializeApp();