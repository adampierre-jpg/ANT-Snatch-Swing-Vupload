/**
 * Version 3.0 — "Lowest Hand Lock" & Robust Physics
 * 
 * - LOGIC: Scans for the "Lowest Hand" (closest to floor) to determine handedness.
 * - FIX: Ignores MediaPipe L/R label swapping near the floor.
 * - FIX: Velocity "Zero Band" (< 0.3m/s) prevents phantom movement from blocking set-end.
 * - UI: Full file upload & camera support restored.
 */

import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const CONFIG = {
  // Landmarks
  LEFT:  { WRIST: 15, SHOULDER: 11, HIP: 23, KNEE: 25, ANKLE: 27 },
  RIGHT: { WRIST: 16, SHOULDER: 12, HIP: 24, KNEE: 26, ANKLE: 28 },

  // Tracking Quality
  MIN_DET_CONF: 0.5,
  MIN_TRACK_CONF: 0.5,

  // Velocity Physics
  SMOOTHING_ALPHA: 0.30,
  MAX_REALISTIC_VELOCITY: 10.0,
  MIN_DT: 0.01,
  MAX_DT: 0.10,
  TORSO_METERS: 0.5,
  ZERO_BAND: 0.3, // Velocity < 0.3 is forced to 0.0

  // Snatch Logic
  LOCKOUT_VY_CUTOFF: 0.40,
  LOCKOUT_SPEED_CUTOFF: 1.40,
  OVERHEAD_HOLD_FRAMES: 2,

  // Drop Feedback
  BASELINE_REPS: 3,
  DROP_WARN: 15,
  DROP_FAIL: 20,

  // Floor/Zone Logic
  MIN_SHANK_LEN_NORM: 0.06,
  
  // Handshake Config
  ARM_MS_REQUIRED: 500,   // 0.5s dwell to start
  RESET_MS_REQUIRED: 500, // 0.5s dwell to end
  STILLNESS_THRESHOLD: 1.5, // m/s (Permissive for jitter, Strict for hiking)
  RESET_GRACE_MS_AFTER_LOCK: 1500 // 1.5s buffer before you can end a set
};

let state = {
  // System
  video: null,
  canvas: null,
  ctx: null,
  landmarker: null,
  isModelLoaded: false,
  isVideoReady: false,
  
  // Runtime
  isTestRunning: false,
  testStage: "IDLE", // IDLE | RUNNING
  
  // Tracking
  lastPose: null,
  lastLoopMs: 0,
  
  // Set Logic
  lockedSide: "unknown",
  armingSide: null, // Candidate side being hovered
  lockedAtMs: 0,
  
  // Dwell Timer (Single shared timer)
  dwellTimerMs: 0,

  // Physics State
  prevWrist: null,
  lockedCalibration: null,
  smoothedVelocity: 0,
  smoothedVy: 0,
  lastSpeed: 0,
  lastVy: 0,

  // Rep Logic
  phase: "IDLE",
  currentRepPeak: 0,
  overheadHoldCount: 0,
  
  // Session Data
  session: {
    currentSet: null, 
    history: []
  },
  
  // Display Helpers
  baseline: 0,
  repHistory: [] 
};

async function initializeApp() {
  state.video = document.getElementById("video");
  state.canvas = document.getElementById("canvas");
  state.ctx = state.canvas.getContext("2d");

  const visionGen = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  state.landmarker = await PoseLandmarker.createFromOptions(visionGen, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: CONFIG.MIN_DET_CONF,
    minTrackingConfidence: CONFIG.MIN_TRACK_CONF
  });

  state.isModelLoaded = true;
  document.getElementById("loading-overlay").classList.add("hidden");
  setStatus("Ready — Upload Video or Start Camera", "#3b82f6");

  // Inputs
  document.getElementById("btn-camera").onclick = startCamera;
  document.getElementById("file-input").onchange = handleUpload;
  document.getElementById("btn-start-test").onclick = toggleTest;
  document.getElementById("btn-reset").onclick = resetSession;
  
  state.video.addEventListener("loadeddata", onVideoReady);
  
  requestAnimationFrame(masterLoop);
}

// --- INPUT HANDLERS ---

function handleUpload(e) {
  const file = e.target.files?.[0];
  if (!file) return;

  resetSession();
  setStatus("Loading Video...", "#fbbf24");
  
  state.video.srcObject = null;
  state.video.src = URL.createObjectURL(file);
  state.video.load();
}

async function startCamera() {
  resetSession();
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false
  });
  state.video.srcObject = stream;
  state.video.src = "";
  state.video.play();
}

function onVideoReady() {
  state.isVideoReady = true;
  state.canvas.width = state.video.videoWidth;
  state.canvas.height = state.video.videoHeight;
  
  document.getElementById("btn-start-test").disabled = false;
  
  // Prime the video
  if (state.video.src) {
      const p = state.video.play();
      if (p && typeof p.then === "function") {
        p.then(() => {
          setTimeout(() => {
            if (!state.isTestRunning) state.video.pause();
            state.video.currentTime = 0;
          }, 100);
        }).catch(() => {});
      }
  }

  setStatus("Video Ready — Press Start Test", "#3b82f6");
}

async function toggleTest() {
  if (!state.isTestRunning) {
    // START
    state.isTestRunning = true;
    state.testStage = "IDLE";
    document.getElementById("btn-start-test").textContent = "Stop Test";
    document.getElementById("btn-reset").disabled = false;
    setStatus("Scanning: Park hand below knee...", "#fbbf24");
    
    if (state.video.paused) {
        try { await state.video.play(); } catch(e) {}
    }
    
  } else {
    // STOP
    state.isTestRunning = false;
    state.testStage = "IDLE";
    document.getElementById("btn-start-test").textContent = "Start Test";
    setStatus("Stopped", "#94a3b8");
    state.video.pause();
  }
}

// --- MASTER LOOP ---

function masterLoop(timestamp) {
  const dt = timestamp - state.lastLoopMs;
  state.lastLoopMs = timestamp;

  if (state.isModelLoaded && state.isVideoReady && state.video.readyState >= 2) {
    const result = state.landmarker.detectForVideo(state.video, timestamp);

    if (result.landmarks && result.landmarks.length > 0) {
      state.lastPose = result.landmarks[0];
      
      // 1. Calculate Velocity (Passive or Active)
      if (state.testStage === "IDLE") {
         calculatePassiveVelocity(state.lastPose, timestamp);
      } else {
         runPhysics(state.lastPose, timestamp);
      }

      // 2. Run Logic
      if (state.isTestRunning) {
        if (state.testStage === "IDLE") {
           checkStartCondition(state.lastPose, dt);
        } else if (state.testStage === "RUNNING") {
           runSnatchLogic(); 
           checkEndCondition(state.lastPose, dt, timestamp);
        }
      }
    }
  }

  drawOverlay();
  requestAnimationFrame(masterLoop);
}

// --- LOGIC: START / END ---

function checkStartCondition(pose, dtMs) {
  // 1. Find LOWEST wrist (Highest Y value)
  const lY = pose[CONFIG.LEFT.WRIST].y;
  const rY = pose[CONFIG.RIGHT.WRIST].y;
  const activeSide = lY > rY ? "left" : "right";
  
  // 2. Check Zone & Stillness
  const inZone = isWristInFloorZone(pose, activeSide);
  const isStill = state.lastSpeed < CONFIG.STILLNESS_THRESHOLD;

  if (inZone && isStill) {
    // Reset timer if we swapped sides suddenly (jitter)
    if (state.armingSide !== activeSide) {
       state.dwellTimerMs = 0;
       state.armingSide = activeSide;
    }
    state.dwellTimerMs += dtMs;
  } else {
    // Reset if moving or out of zone
    state.dwellTimerMs = 0;
    state.armingSide = null;
  }

  // 3. Trigger Lock
  if (state.dwellTimerMs >= CONFIG.ARM_MS_REQUIRED) {
    startNewSet(state.armingSide);
  }
}

function checkEndCondition(pose, dtMs, totalTimeMs) {
  // Grace Period
  if (totalTimeMs - state.lockedAtMs < CONFIG.RESET_GRACE_MS_AFTER_LOCK) return;

  const inZone = isWristInFloorZone(pose, state.lockedSide);
  const isStill = state.lastSpeed < CONFIG.STILLNESS_THRESHOLD;

  if (inZone && isStill) {
    state.dwellTimerMs += dtMs;
  } else {
    state.dwellTimerMs = 0;
  }

  if (state.dwellTimerMs >= CONFIG.RESET_MS_REQUIRED) {
    endCurrentSet();
  }
}

function startNewSet(side) {
  state.lockedSide = side;
  state.testStage = "RUNNING";
  state.lockedAtMs = state.lastLoopMs;
  state.dwellTimerMs = 0;
  
  // Clean Slate
  state.repHistory = [];
  state.currentRepPeak = 0;
  state.overheadHoldCount = 0;
  state.phase = "IDLE";
  
  // Session
  state.session.currentSet = {
    id: state.session.history.length + 1,
    hand: side,
    reps: [],
    startTime: new Date()
  };

  updateUIValues(0, 0, "--", "#fff");
  setStatus(`LOCKED: ${side.toUpperCase()}`, "#10b981");
}

function endCurrentSet() {
  if (state.session.currentSet) {
    state.session.currentSet.endTime = new Date();
    state.session.history.push(state.session.currentSet);
  }
  
  state.testStage = "IDLE";
  state.lockedSide = "unknown";
  state.dwellTimerMs = 0;
  state.session.currentSet = null;

  setStatus("Set Saved. Park to start next.", "#3b82f6");
}

// --- PHYSICS ENGINE ---

function isWristInFloorZone(pose, side) {
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const knee = pose[idx.KNEE];
  
  if (!wrist || !knee) return false;
  if (wrist.y < 0 || wrist.y > 1.0) return false; // Sanity check

  return wrist.y > knee.y; 
}

function calculatePassiveVelocity(pose, timeMs) {
  // Always track the LOWEST hand for IDLE velocity
  const lY = pose[CONFIG.LEFT.WRIST].y;
  const rY = pose[CONFIG.RIGHT.WRIST].y;
  const activeSide = lY > rY ? "left" : "right";
  
  const oldSide = state.lockedSide;
  state.lockedSide = activeSide;
  runPhysics(pose, timeMs);
  state.lockedSide = oldSide; 
}

function runPhysics(pose, timeMs) {
  const side = state.lockedSide;
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  
  const wrist = pose[idx.WRIST];
  const shoulder = pose[idx.SHOULDER];
  const hip = pose[idx.HIP];
  if (!wrist || !shoulder || !hip) return;

  // Calibration
  if (!state.lockedCalibration) {
    const torsoPx = Math.abs(shoulder.y - hip.y) * state.canvas.height;
    state.lockedCalibration = torsoPx > 0 ? torsoPx / CONFIG.TORSO_METERS : 100;
  }

  // DT
  if (!state.prevWrist) {
    state.prevWrist = { x: wrist.x, y: wrist.y, t: timeMs };
    return;
  }
  const dt = (timeMs - state.prevWrist.t) / 1000;
  if (dt < CONFIG.MIN_DT || dt > CONFIG.MAX_DT) {
    state.prevWrist = { x: wrist.x, y: wrist.y, t: timeMs };
    return;
  }

  // Velocity
  const dxPx = (wrist.x - state.prevWrist.x) * state.canvas.width;
  const dyPx = (wrist.y - state.prevWrist.y) * state.canvas.height;

  const vx = (dxPx / state.lockedCalibration) / dt;
  const vy = (dyPx / state.lockedCalibration) / dt;
  let speed = Math.hypot(vx, vy);

  // ZERO BAND (Fix for phantom drift)
  if (speed < CONFIG.ZERO_BAND) {
    speed = 0;
  }

  // Smoothing
  state.smoothedVelocity = CONFIG.SMOOTHING_ALPHA * speed + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVelocity;
  state.smoothedVy = CONFIG.SMOOTHING_ALPHA * vy + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVy;
  
  state.lastSpeed = state.smoothedVelocity;
  state.lastVy = state.smoothedVy;
  
  state.prevWrist = { x: wrist.x, y: wrist.y, t: timeMs };
  
  if (state.testStage === "RUNNING") {
    document.getElementById("val-velocity").textContent = state.smoothedVelocity.toFixed(2);
  }
}

function runSnatchLogic() {
  const v = state.smoothedVelocity;
  const vy = state.smoothedVy;
  const pose = state.lastPose;
  const idx = state.lockedSide === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const shoulder = pose[idx.SHOULDER];
  const hip = pose[idx.HIP];

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
    }
  } else if (state.phase === "CONCENTRIC") {
    if (v > state.currentRepPeak) state.currentRepPeak = v;

    const lockoutOk = isAboveShoulder && Math.abs(vy) < CONFIG.LOCKOUT_VY_CUTOFF && v < CONFIG.LOCKOUT_SPEED_CUTOFF;
    
    if (lockoutOk) {
      state.overheadHoldCount++;
      if (state.overheadHoldCount >= CONFIG.OVERHEAD_HOLD_FRAMES) {
        recordRep();
      }
    } else {
      state.overheadHoldCount = 0;
      if (isBelowHip) state.phase = "BOTTOM";
    }
  }
}

function recordRep() {
  state.phase = "LOCKOUT";
  state.overheadHoldCount = 0;
  
  if (state.session.currentSet) state.session.currentSet.reps.push(state.currentRepPeak);
  
  state.repHistory.push(state.currentRepPeak);
  const repCount = state.repHistory.length;
  
  let dropText = "--";
  let dropColor = "#10b981";
  
  if (repCount <= CONFIG.BASELINE_REPS) {
    state.baseline = state.repHistory.reduce((a,b)=>a+b,0) / repCount;
    dropText = "CALC";
  } else {
    const drop = (state.baseline - state.currentRepPeak) / state.baseline;
    const dropPct = (drop * 100).toFixed(1);
    dropText = `-${dropPct}%`;
    if (drop * 100 >= CONFIG.DROP_FAIL) dropColor = "#ef4444";
    else if (drop * 100 >= CONFIG.DROP_WARN) dropColor = "#fbbf24";
  }

  updateUIValues(repCount, state.currentRepPeak, dropText, dropColor);
}

// --- UI UTILS ---

function updateUIValues(reps, peak, drop, dropColor) {
  document.getElementById("val-reps").textContent = reps;
  document.getElementById("val-peak").textContent = peak.toFixed(2);
  const d = document.getElementById("val-drop");
  d.textContent = drop;
  if (dropColor) d.style.color = dropColor;
}

function resetSession() {
  state.session = { currentSet: null, history: [] };
  state.repHistory = [];
  state.baseline = 0;
  state.testStage = "IDLE";
  state.lockedSide = "unknown";
  state.dwellTimerMs = 0;
  updateUIValues(0, 0, "--", "#fff");
  setStatus("Reset Complete", "#3b82f6");
}

function setStatus(text, color) {
  const pill = document.getElementById("status-pill");
  if (pill) {
    pill.textContent = text;
    pill.style.color = color;
    pill.style.borderColor = color;
  }
}

function drawOverlay() {
  state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
  if (!state.lastPose) return;
  
  // Debug Text (Remove later if needed)
  state.ctx.fillStyle = "#fbbf24";
  state.ctx.font = "14px monospace";
  state.ctx.fillText(`Timer: ${(state.dwellTimerMs/1000).toFixed(1)}s`, 10, 20);
  state.ctx.fillText(`Speed: ${state.lastSpeed.toFixed(2)}`, 10, 40);

  const side = state.lockedSide;

  if (state.testStage === "RUNNING") {
    const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
    const wrist = state.lastPose[idx.WRIST];
    drawDot(wrist, true, "#10b981");
    drawParkingLine(state.lastPose, side, state.lastSpeed < CONFIG.STILLNESS_THRESHOLD);
  } else {
    // IDLE: Draw lowest hand emphasized
    const lY = state.lastPose[CONFIG.LEFT.WRIST].y;
    const rY = state.lastPose[CONFIG.RIGHT.WRIST].y;
    const lowest = lY > rY ? "left" : "right";
    
    const isStill = state.lastSpeed < CONFIG.STILLNESS_THRESHOLD;
    const color = isStill ? "#fbbf24" : "#94a3b8"; 

    drawDot(state.lastPose[CONFIG.LEFT.WRIST], lowest==="left", color);
    drawDot(state.lastPose[CONFIG.RIGHT.WRIST], lowest==="right", color);
    
    drawParkingLine(state.lastPose, "left", false);
    drawParkingLine(state.lastPose, "right", false);
  }
}

function drawParkingLine(pose, side, isActive) {
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const knee = pose[idx.KNEE];
  if (!knee) return;
  
  const y = knee.y * state.canvas.height;
  state.ctx.beginPath();
  state.ctx.strokeStyle = isActive ? "#10b981" : "rgba(255,255,255,0.2)";
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

initializeApp();
