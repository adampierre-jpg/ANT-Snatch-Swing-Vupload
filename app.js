/**
 * Version C — "Strict Parking Logic"
 * - Start: Wrist below knee + Speed < 0.5 + 0.3s dwell.
 * - End: Wrist below knee + Speed < 0.5 + 0.3s dwell.
 * - Prevents "Hike Trap" (ending set mid-swing) by enforcing stillness.
 * - Handles Set/Session management.
 * 
 * Logic Sources:
 * - MediaPipe detectForVideo: Monotonic ms timestamps [web:114].
 * - Landmarks: Wrist(15/16), Knee(25/26) [web:53].
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

  // Snatch Logic
  LOCKOUT_VY_CUTOFF: 0.40,
  LOCKOUT_SPEED_CUTOFF: 1.40,
  OVERHEAD_HOLD_FRAMES: 2,

  // Drop Feedback
  BASELINE_REPS: 3,
  DROP_WARN: 15,
  DROP_FAIL: 20,

  // Floor/Zone Logic
  SHANK_FRACTION: 0.35,
  MIN_SHANK_LEN_NORM: 0.06,
  
  // 0.3s Handshake Config
  ARM_MS_REQUIRED: 300,   // 0.3 seconds to start
  RESET_MS_REQUIRED: 300, // 0.3 seconds to end
  STILLNESS_THRESHOLD: 0.5, // m/s (Must be still to trigger start/end)
  RESET_GRACE_MS_AFTER_LOCK: 1000 // Buffer so you don't unlock instantly on pickup
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
  lockedAtMs: 0,
  
  // Dwell Timers (for 0.3s logic)
  floorMsLeft: 0,
  floorMsRight: 0,
  floorMsLocked: 0,

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
    currentSet: null, // { id: 1, hand: 'left', reps: [], avgPeak: 0 }
    history: []
  },
  
  // Display Helpers
  baseline: 0,
  repHistory: [] // Local history for the current active set
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
  setStatus("Ready — Press Start", "#3b82f6");

  document.getElementById("btn-camera").onclick = startCamera;
  document.getElementById("btn-start-test").onclick = toggleTest;
  document.getElementById("btn-reset").onclick = resetSession; // Full wipe
  
  state.video.addEventListener("loadeddata", onVideoReady);
  
  requestAnimationFrame(masterLoop);
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false
  });
  state.video.srcObject = stream;
  state.video.play();
}

function onVideoReady() {
  state.isVideoReady = true;
  state.canvas.width = state.video.videoWidth;
  state.canvas.height = state.video.videoHeight;
  document.getElementById("btn-start-test").disabled = false;
  setStatus("Camera Ready — Start Test", "#3b82f6");
}

function toggleTest() {
  if (!state.isTestRunning) {
    state.isTestRunning = true;
    state.testStage = "IDLE";
    document.getElementById("btn-start-test").textContent = "Stop Test";
    document.getElementById("btn-reset").disabled = false;
    setStatus("Scanning: Park hand below knee...", "#fbbf24");
  } else {
    state.isTestRunning = false;
    state.testStage = "IDLE";
    document.getElementById("btn-start-test").textContent = "Start Test";
    setStatus("Stopped", "#94a3b8");
  }
}

function masterLoop(timestamp) {
  // timestamp is monotonic (performance.now())
  const dt = timestamp - state.lastLoopMs;
  state.lastLoopMs = timestamp;

  if (state.isModelLoaded && state.isVideoReady && state.video.readyState >= 2) {
    // Detect pose
    const result = state.landmarker.detectForVideo(state.video, timestamp);

    if (result.landmarks && result.landmarks.length > 0) {
      state.lastPose = result.landmarks[0];
      
      // We always run physics if locked, to get velocity for the "Stillness Check"
      if (state.testStage === "RUNNING") {
        runPhysics(state.lastPose, timestamp);
      } else if (state.testStage === "IDLE") {
        // Run physics on BOTH hands loosely just to get 'lastSpeed' for arming check?
        // Actually, we can approximate speed for arming logic inside the arming function
        // or just rely on position for the first frame.
        // Better: We need velocity to arm safely. 
        // Let's run a "passive" physics check on the closest hand to floor.
        calculatePassiveVelocity(state.lastPose, timestamp);
      }

      if (state.isTestRunning) {
        if (state.testStage === "IDLE") {
           checkStartCondition(state.lastPose, dt);
        } else if (state.testStage === "RUNNING") {
           runSnatchLogic(); // Rep counting
           checkEndCondition(state.lastPose, dt, timestamp);
        }
      }
    }
  }

  drawOverlay();
  requestAnimationFrame(masterLoop);
}

// --- 0.3s LOGIC FUNCTIONS ---

function checkStartCondition(pose, dtMs) {
  // 1. Check Zones
  const inL = isWristInFloorZone(pose, "left");
  const inR = isWristInFloorZone(pose, "right");

  // 2. Check Stillness (Must be < 0.5 m/s)
  // We use state.lastSpeed which is updated by calculatePassiveVelocity
  const isStill = state.lastSpeed < CONFIG.STILLNESS_THRESHOLD;

  // 3. Increment Timers
  if (inL && isStill) {
    state.floorMsLeft += dtMs;
    state.floorMsRight = 0;
  } else if (inR && isStill) {
    state.floorMsRight += dtMs;
    state.floorMsLeft = 0;
  } else {
    // Reset if moving or out of zone
    state.floorMsLeft = 0;
    state.floorMsRight = 0;
  }

  // 4. Trigger Lock (0.3s)
  if (state.floorMsLeft >= CONFIG.ARM_MS_REQUIRED) {
    startNewSet("left");
  } else if (state.floorMsRight >= CONFIG.ARM_MS_REQUIRED) {
    startNewSet("right");
  }
}

function checkEndCondition(pose, dtMs, totalTimeMs) {
  // Buffer: Don't allow ending immediately after starting (e.g. 1s grace)
  if (totalTimeMs - state.lockedAtMs < CONFIG.RESET_GRACE_MS_AFTER_LOCK) return;

  const inZone = isWristInFloorZone(pose, state.lockedSide);
  
  // Crucial: HIKING has high speed. PARKING has low speed.
  const isStill = state.lastSpeed < CONFIG.STILLNESS_THRESHOLD;

  if (inZone && isStill) {
    state.floorMsLocked += dtMs;
  } else {
    state.floorMsLocked = 0;
  }

  if (state.floorMsLocked >= CONFIG.RESET_MS_REQUIRED) {
    endCurrentSet();
  }
}

function startNewSet(side) {
  state.lockedSide = side;
  state.testStage = "RUNNING";
  state.lockedAtMs = state.lastLoopMs;
  
  // RESET Physics/Rep Counters for this new set
  state.repHistory = [];
  state.currentRepPeak = 0;
  state.overheadHoldCount = 0;
  state.phase = "IDLE";
  
  // Create Session Object
  state.session.currentSet = {
    id: state.session.history.length + 1,
    hand: side,
    reps: [],
    startTime: new Date()
  };

  // Update UI
  updateUIValues(0, 0, "--");
  setStatus(`LOCKED: ${side.toUpperCase()}`, "#10b981");
}

function endCurrentSet() {
  // Save to history
  if (state.session.currentSet) {
    state.session.currentSet.endTime = new Date();
    state.session.history.push(state.session.currentSet);
  }
  
  state.testStage = "IDLE";
  state.lockedSide = "unknown";
  state.floorMsLeft = 0;
  state.floorMsRight = 0;
  state.floorMsLocked = 0;
  state.session.currentSet = null;

  setStatus("Set Saved. Park to start next.", "#3b82f6");
}

// --- PHYSICS & LOGIC ---

function isWristInFloorZone(pose, side) {
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const knee = pose[idx.KNEE];
  const ankle = pose[idx.ANKLE];
  if (!wrist || !knee || !ankle) return false;

  const shank = ankle.y - knee.y;
  // Guard against sitting/weird angles
  if (shank < CONFIG.MIN_SHANK_LEN_NORM) return false; 

  // Definition: Wrist below the top of the knee
  // (In normalized coords, larger Y is lower)
  return wrist.y > knee.y; 
}

function calculatePassiveVelocity(pose, timeMs) {
  // Just track the lowest hand to gauge "general stillness"
  const lY = pose[CONFIG.LEFT.WRIST].y;
  const rY = pose[CONFIG.RIGHT.WRIST].y;
  
  // Pick the hand closer to floor (higher Y value)
  const activeSide = lY > rY ? "left" : "right";
  
  // Reuse the main physics function but don't count reps
  // Hack: temporarily set lockedSide so physics can run math
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

  // Calibration (Torso Scale)
  if (!state.lockedCalibration) {
    const torsoPx = Math.abs(shoulder.y - hip.y) * state.canvas.height;
    state.lockedCalibration = torsoPx > 0 ? torsoPx / CONFIG.TORSO_METERS : 100;
  }

  // Delta Time
  if (!state.prevWrist) {
    state.prevWrist = { x: wrist.x, y: wrist.y, t: timeMs };
    return;
  }
  
  const dt = (timeMs - state.prevWrist.t) / 1000;
  if (dt < CONFIG.MIN_DT || dt > CONFIG.MAX_DT) {
    state.prevWrist = { x: wrist.x, y: wrist.y, t: timeMs };
    return;
  }

  // Velocity Calculation
  const dxPx = (wrist.x - state.prevWrist.x) * state.canvas.width;
  const dyPx = (wrist.y - state.prevWrist.y) * state.canvas.height;

  const vx = (dxPx / state.lockedCalibration) / dt;
  const vy = (dyPx / state.lockedCalibration) / dt;
  const speed = Math.hypot(vx, vy);

  // Smoothing
  state.smoothedVelocity = CONFIG.SMOOTHING_ALPHA * speed + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVelocity;
  state.smoothedVy = CONFIG.SMOOTHING_ALPHA * vy + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVy;
  
  // Store for Logic
  state.lastSpeed = state.smoothedVelocity;
  state.lastVy = state.smoothedVy;
  
  state.prevWrist = { x: wrist.x, y: wrist.y, t: timeMs };
  
  // Live Update (only if running)
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

  // State Machine
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
  
  // Add to Session
  if (state.session.currentSet) {
    state.session.currentSet.reps.push(state.currentRepPeak);
  }
  
  state.repHistory.push(state.currentRepPeak);
  const repCount = state.repHistory.length;
  
  // Calc Drop
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

// --- UTILS ---

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

// --- VISUALS ---

function drawOverlay() {
  state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
  if (!state.lastPose) return;
  
  const side = state.lockedSide;

  if (state.testStage === "RUNNING") {
    // Draw Active Hand
    const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
    const wrist = state.lastPose[idx.WRIST];
    drawDot(wrist, true, "#10b981");
    
    // Draw Parking Zone (Visual feedback for stopping)
    drawParkingLine(state.lastPose, side, state.lastSpeed < CONFIG.STILLNESS_THRESHOLD);
  
  } else {
    // IDLE: Draw both hands
    const L = state.lastPose[CONFIG.LEFT.WRIST];
    const R = state.lastPose[CONFIG.RIGHT.WRIST];
    
    // Color them based on 'ready' (stillness)
    const isStill = state.lastSpeed < CONFIG.STILLNESS_THRESHOLD;
    const color = isStill ? "#fbbf24" : "#94a3b8"; 
    
    drawDot(L, false, color);
    drawDot(R, false, color);
    
    // Draw parking lines for both
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
