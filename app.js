/**
 * Version 3.4 --- MOVEMENT AUTO-DETECTION ENABLED
 * - Added: Multi-movement detection (Swing, Clean, Snatch)
 * - Added: Configuration detection (Single, Two-hands, Double bells)
 * - Preserves: All v3.3 snatch tracking functionality
 */

import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

// ============================================
// CONFIG (Version 3.4 - Movement Detection Added)
// ============================================

const CONFIG = {
  // LANDMARKS (MediaPipe Body Pose)
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

  // VELOCITY PHYSICS (Stabilized)
  SMOOTHING_ALPHA: 0.15,
  MAX_REALISTIC_VELOCITY: 8.0,
  ZERO_BAND: 0.1,
  MIN_DT: 0.016,
  MAX_DT: 0.1,

  // LOCKOUT DETECTION (Balanced)
  LOCKOUT_VY_CUTOFF: 0.6,
  LOCKOUT_SPEED_CUTOFF: 2.0,

  // START/STOP GESTURES
  RESET_GRACE_MS_AFTER_LOCK: 2000,
  HIKE_VY_THRESHOLD: 0.3,
  HIKE_SPEED_THRESHOLD: 0.5,

  // REP & DROP-OFF LOGIC
  BASELINE_REPS: 3,
  DROP_WARN: 15,
  DROP_FAIL: 20,

  // MEDIAPIPE SETTINGS
  MIN_DET_CONF: 0.5,
  MIN_TRACK_CONF: 0.5,

  // ✅ NEW: MOVEMENT DETECTION THRESHOLDS
  DETECTION: {
    WRIST_PROXIMITY_THRESHOLD: 0.15,    // Two hands on one bell
    DOUBLE_BELL_SPACING: 0.25,          // Double kettlebells
    ACTIVE_VELOCITY_THRESHOLD: 0.8,     // Active movement
    INACTIVE_VELOCITY_THRESHOLD: 0.3,   // Parked/inactive
    SWING_HEIGHT_THRESHOLD: 0.08,       // Below shoulder + threshold = swing
    RACK_HEIGHT_THRESHOLD: 0.08,        // Shoulder ± threshold = clean
    OVERHEAD_HEIGHT_THRESHOLD: 0.1,     // Well above shoulder = snatch
    RACK_HORIZONTAL_PROXIMITY: 0.15     // Distance from torso for clean
  },

  // EXPORT
  MAKE_WEBHOOK_URL: "https://hook.us2.make.com/0l88dnosrk2t8a29yfk83fej8hp8j3jk",

  // DEBUGGING
  DEBUG_MODE: true
};

// ============================================
// STATE (Extended for Movement Detection)
// ============================================

let state = {
  video: null,
  canvas: null,
  ctx: null,
  landmarker: null,
  isModelLoaded: false,
  isVideoReady: false,
  isTestRunning: false,
  testStage: "IDLE",
  timeMs: 0,
  lastPose: null,

  // Hand Tracking
  activeTrackingSide: "left",
  lockedSide: "unknown",
  armingSide: null,

  // Physics (SHARED - used for active hand)
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
  phase: "IDLE",
  currentRepPeak: 0,
  overheadHoldCount: 0,
  session: {
    currentSet: null,
    history: []
  },
  repHistory: [],
  baseline: 0,

  // ✅ NEW: Movement Detection State
  movementDetection: {
    leftWristVelocity: 0,
    rightWristVelocity: 0,
    leftPrevWrist: null,
    rightPrevWrist: null,
    wristSpacing: 0,
    detectedMovement: 'UNKNOWN',
    detectedConfig: 'UNKNOWN',
    detectedHands: 'none',
    movementHistory: []
  }
};

// ============================================
// INITIALIZATION
// ============================================

async function initializeApp() {
  state.video = document.getElementById("video");
  state.canvas = document.getElementById("canvas");
  state.ctx = state.canvas.getContext("2d");

  document.getElementById("btn-camera").onclick = startCamera;
  document.getElementById("file-input").onchange = handleUpload;
  document.getElementById("btn-start-test").onclick = toggleTest;
  document.getElementById("btn-reset").onclick = resetSession;

  const saveBtn = document.getElementById("btn-save");
  if (saveBtn) saveBtn.onclick = exportToMake;

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
  setStatus("Vision System Online — Upload Video or Start Camera", "#3b82f6");

  state.video.addEventListener("loadeddata", onVideoReady);
  requestAnimationFrame(masterLoop);
}

// ============================================
// VIDEO INPUTS
// ============================================

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

  setStatus("Activating Optics...", "#fbbf24");
  state.video.srcObject = stream;
  state.video.src = "";
  state.video.play();
}

function onVideoReady() {
  state.isVideoReady = true;
  state.canvas.width = state.video.videoWidth;
  state.canvas.height = state.video.videoHeight;

  document.getElementById("btn-start-test").disabled = false;

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
    state.isTestRunning = true;
    state.testStage = "IDLE";
    document.getElementById("btn-start-test").textContent = "Stop Test";
    document.getElementById("btn-reset").disabled = false;
    setStatus("Scanning: Park hand below knee...", "#fbbf24");

    // ✅ NEW: Reset movement detection UI
    resetMovementDisplay();

    if (state.video.paused) {
      try { await state.video.play(); } catch(e) {}
    }
  } else {
    state.isTestRunning = false;
    state.testStage = "IDLE";
    document.getElementById("btn-start-test").textContent = "Start Test";
    setStatus("Stopped", "#94a3b8");
    state.video.pause();
  }
}

// ============================================
// MASTER LOOP (Enhanced with Movement Detection)
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
      console.warn("Detection error:", e);
    }
  }

  if (!pose) {
    drawOverlay();
    return;
  }

  // ✅ ALWAYS RUN PHYSICS (even in IDLE)
  if (state.isTestRunning) {
    runPhysics(pose, state.timeMs);

    // ✅ NEW: Run dual-wrist tracking in parallel
    runDualWristPhysics(pose, state.timeMs);

    // ✅ NEW: Update configuration detection (only after calibration)
    if (state.lockedCalibration) {
      detectMovementConfiguration();
    }

    if (state.testStage === "IDLE") {
      checkStartCondition(pose, state.timeMs);
    }

    if (state.testStage === "RUNNING") {
      runSnatchLogic(pose);
      checkEndCondition(pose, state.timeMs);
    }
  }

  drawOverlay();
}

// ============================================
// START/END CONDITIONS
// ============================================

function checkStartCondition(pose, timeMs) {
  if (state.testStage !== "IDLE") return;

  const lWrist = pose[CONFIG.LEFT.WRIST];
  const rWrist = pose[CONFIG.RIGHT.WRIST];
  if (!lWrist || !rWrist) return;

  const lY = lWrist.y;
  const rY = rWrist.y;
  const activeSide = lY > rY ? "left" : "right";

  // Update tracking side
  state.activeTrackingSide = activeSide;

  const inZone = isWristInFloorZone(pose, activeSide);
  const hikingDown = state.lastVy > 0.3 && state.lastSpeed > 0.5;

  // ✅ DEBUG LOG
  if (CONFIG.DEBUG_MODE && inZone) {
    console.log(`[START] Side:${activeSide} | Zone:${inZone} | Hike:${hikingDown} | Vy:${state.lastVy.toFixed(2)} | Speed:${state.lastSpeed.toFixed(2)}`);
  }

  // ✅ SIMPLE: Just zone + hike direction
  if (inZone && hikingDown) {
    console.log(`🚀 STARTING SET: ${activeSide}`);
    startNewSet(activeSide);
  }
}

function checkEndCondition(pose, timeMs) {
  if (state.testStage !== "RUNNING") return;
  if (!state.session.currentSet) return;

  const grace = timeMs - state.session.currentSet.lockedAtMs;
  if (grace < CONFIG.RESET_GRACE_MS_AFTER_LOCK) return;

  const sideIdx = state.lockedSide === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[sideIdx.WRIST];
  if (!wrist) return;

  const inZone = isWristInFloorZone(pose, state.lockedSide);
  const standingUp = state.lastVy < -0.3 && state.lastSpeed > 0.5;

  // ✅ DEBUG LOG
  if (CONFIG.DEBUG_MODE && inZone) {
    console.log(`[END] Zone:${inZone} | StandUp:${standingUp} | Vy:${state.lastVy.toFixed(2)} | Speed:${state.lastSpeed.toFixed(2)}`);
  }

  // ✅ SIMPLE: Just zone + upward direction
  if (inZone && standingUp) {
    console.log(`🛑 ENDING SET`);
    endCurrentSet();
  }
}

// ============================================
// SET MANAGEMENT
// ============================================

function startNewSet(side) {
  state.lockedSide = side;
  state.testStage = "RUNNING";
  state.lockedAtMs = state.timeMs;

  // Reset metrics for the new set
  state.repHistory = [];
  state.currentRepPeak = 0;
  state.overheadHoldCount = 0;
  state.baseline = 0;

  // Pre-prime the phase to BOTTOM
  state.phase = "BOTTOM";

  // Initialize the session object for export
  state.session.currentSet = {
    id: state.session.history.length + 1,
    hand: side,
    reps: [],
    startTime: new Date(),
    lockedAtMs: state.timeMs
  };

  // Update UI
  updateUIValues(0, 0, "--", "#fff");
  setStatus(`LOCKED: ${side.toUpperCase()}`, "#10b981");

  if (CONFIG.DEBUG_MODE) console.log(`🚀 Set Started [${side}]. Phase: BOTTOM.`);
}

function endCurrentSet() {
  if (state.session.currentSet) {
    state.session.currentSet.endTime = new Date();
    const peaks = state.session.currentSet.reps;

    // ✅ BACKWARD COMPATIBLE: Handle both number and object formats
    const velocities = peaks.map(rep => typeof rep === 'number' ? rep : rep.velocity);
    const avg = velocities.length > 0 ? velocities.reduce((a,b)=>a+b,0)/velocities.length : 0;

    state.session.currentSet.avgVelocity = avg.toFixed(2);
    state.session.history.push(state.session.currentSet);
  }

  state.testStage = "IDLE";
  state.lockedSide = "unknown";
  state.session.currentSet = null;
  state.activeTrackingSide = "left";

  setStatus("Set Saved. Park to start next.", "#3b82f6");
}

// ============================================
// PHYSICS ENGINE (Original - Unchanged)
// ============================================

function runPhysics(pose, timeMs) {
  // IN IDLE: Track the lower hand
  // IN RUNNING: Track the locked side
  const side = state.testStage === "IDLE" ? state.activeTrackingSide : state.lockedSide;
  if (!side || side === "unknown") return;

  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const shoulder = pose[idx.SHOULDER];
  const hip = pose[idx.HIP];
  if (!wrist || !shoulder || !hip) return;

  // Calibration
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

  // Frame normalization
  const TARGET_FPS = 30;
  const frameTimeMs = 1000 / TARGET_FPS;
  const actualFrameTimeMs = timeMs - state.prevWrist.t;
  const timeRatio = frameTimeMs / actualFrameTimeMs;
  vx *= timeRatio;
  vy *= timeRatio;
  speed = Math.hypot(vx, vy);

  // Zero band
  if (speed < CONFIG.ZERO_BAND) speed = 0;

  // Smoothing
  state.smoothedVelocity = CONFIG.SMOOTHING_ALPHA * speed + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVelocity;
  state.smoothedVy = CONFIG.SMOOTHING_ALPHA * vy + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVy;

  // Ceiling
  state.lastSpeed = Math.min(state.smoothedVelocity, CONFIG.MAX_REALISTIC_VELOCITY);
  state.lastVy = Math.min(Math.max(state.smoothedVy, -CONFIG.MAX_REALISTIC_VELOCITY), CONFIG.MAX_REALISTIC_VELOCITY);

  state.prevWrist = { x: wrist.x, y: wrist.y, t: timeMs };

  // Update UI
  if (state.testStage === "RUNNING") {
    document.getElementById("val-velocity").textContent = state.lastSpeed.toFixed(2);
  }
}

// ============================================
// ✅ NEW: DUAL-WRIST PHYSICS (Parallel Tracking)
// ============================================

function runDualWristPhysics(pose, timeMs) {
  if (state.testStage !== "IDLE" && state.testStage !== "RUNNING") return;

  const lWrist = pose[CONFIG.LEFT.WRIST];
  const rWrist = pose[CONFIG.RIGHT.WRIST];
  const lShoulder = pose[CONFIG.LEFT.SHOULDER];
  const rShoulder = pose[CONFIG.RIGHT.SHOULDER];
  const lHip = pose[CONFIG.LEFT.HIP];
  const rHip = pose[CONFIG.RIGHT.HIP];

  if (!lWrist || !rWrist || !lShoulder || !rShoulder || !lHip || !rHip) return;

  // Use existing calibration from main physics engine
  if (!state.lockedCalibration) return;

  // Track LEFT wrist velocity
  if (!state.movementDetection.leftPrevWrist) {
    state.movementDetection.leftPrevWrist = { x: lWrist.x, y: lWrist.y, t: timeMs };
  } else {
    const dt = (timeMs - state.movementDetection.leftPrevWrist.t) / 1000;
    if (dt >= CONFIG.MIN_DT && dt <= CONFIG.MAX_DT) {
      const dxPx = (lWrist.x - state.movementDetection.leftPrevWrist.x) * state.canvas.width;
      const dyPx = (lWrist.y - state.movementDetection.leftPrevWrist.y) * state.canvas.height;
      let vx = (dxPx / state.lockedCalibration) / dt;
      let vy = (dyPx / state.lockedCalibration) / dt;

      // FPS normalization
      const TARGET_FPS = 30;
      const frameTimeMs = 1000 / TARGET_FPS;
      const actualFrameTimeMs = timeMs - state.movementDetection.leftPrevWrist.t;
      const timeRatio = frameTimeMs / actualFrameTimeMs;
      vx *= timeRatio;
      vy *= timeRatio;

      let speed = Math.hypot(vx, vy);
      if (speed < CONFIG.ZERO_BAND) speed = 0;

      // Light smoothing for detection
      state.movementDetection.leftWristVelocity = 0.3 * speed + 0.7 * state.movementDetection.leftWristVelocity;
    }
    state.movementDetection.leftPrevWrist = { x: lWrist.x, y: lWrist.y, t: timeMs };
  }

  // Track RIGHT wrist velocity
  if (!state.movementDetection.rightPrevWrist) {
    state.movementDetection.rightPrevWrist = { x: rWrist.x, y: rWrist.y, t: timeMs };
  } else {
    const dt = (timeMs - state.movementDetection.rightPrevWrist.t) / 1000;
    if (dt >= CONFIG.MIN_DT && dt <= CONFIG.MAX_DT) {
      const dxPx = (rWrist.x - state.movementDetection.rightPrevWrist.x) * state.canvas.width;
      const dyPx = (rWrist.y - state.movementDetection.rightPrevWrist.y) * state.canvas.height;
      let vx = (dxPx / state.lockedCalibration) / dt;
      let vy = (dyPx / state.lockedCalibration) / dt;

      const TARGET_FPS = 30;
      const frameTimeMs = 1000 / TARGET_FPS;
      const actualFrameTimeMs = timeMs - state.movementDetection.rightPrevWrist.t;
      const timeRatio = frameTimeMs / actualFrameTimeMs;
      vx *= timeRatio;
      vy *= timeRatio;

      let speed = Math.hypot(vx, vy);
      if (speed < CONFIG.ZERO_BAND) speed = 0;

      state.movementDetection.rightWristVelocity = 0.3 * speed + 0.7 * state.movementDetection.rightWristVelocity;
    }
    state.movementDetection.rightPrevWrist = { x: rWrist.x, y: rWrist.y, t: timeMs };
  }

  // Calculate wrist spacing (normalized 0-1)
  state.movementDetection.wristSpacing = Math.abs(lWrist.x - rWrist.x);
}

// ============================================
// ✅ NEW: MOVEMENT CONFIGURATION DETECTION
// ============================================

function detectMovementConfiguration() {
  const lVel = state.movementDetection.leftWristVelocity;
  const rVel = state.movementDetection.rightWristVelocity;
  const spacing = state.movementDetection.wristSpacing;

  const leftActive = lVel > CONFIG.DETECTION.ACTIVE_VELOCITY_THRESHOLD;
  const rightActive = rVel > CONFIG.DETECTION.ACTIVE_VELOCITY_THRESHOLD;

  let config = 'UNKNOWN';
  let hands = 'none';

  // Both wrists moving
  if (leftActive && rightActive) {
    if (spacing < CONFIG.DETECTION.WRIST_PROXIMITY_THRESHOLD) {
      config = 'TWO_HANDS_ONE_BELL';
      hands = 'both';
    } else if (spacing > CONFIG.DETECTION.DOUBLE_BELL_SPACING) {
      config = 'DOUBLE_BELLS';
      hands = 'both';
    }
  }
  // Only left moving
  else if (leftActive && !rightActive) {
    config = 'SINGLE_BELL';
    hands = 'left';
  }
  // Only right moving
  else if (rightActive && !leftActive) {
    config = 'SINGLE_BELL';
    hands = 'right';
  }

  state.movementDetection.detectedConfig = config;
  state.movementDetection.detectedHands = hands;

  return { config, hands };
}

// ============================================
// ✅ NEW: MOVEMENT TYPE CLASSIFICATION
// ============================================

function classifyMovementAtLockout() {
  if (!state.lastPose) return 'UNKNOWN';

  const idx = state.lockedSide === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = state.lastPose[idx.WRIST];
  const shoulder = state.lastPose[idx.SHOULDER];
  const hip = state.lastPose[idx.HIP];

  if (!wrist || !shoulder || !hip) return 'UNKNOWN';

  const config = state.movementDetection.detectedConfig;
  const hands = state.movementDetection.detectedHands;

  // Determine movement type based on peak height
  let movementType = 'UNKNOWN';

  // Check peak position during lockout
  const isOverhead = wrist.y < (shoulder.y - CONFIG.DETECTION.OVERHEAD_HEIGHT_THRESHOLD);
  const isRackZone = wrist.y >= (shoulder.y - CONFIG.DETECTION.RACK_HEIGHT_THRESHOLD) && 
                      wrist.y < (shoulder.y + CONFIG.DETECTION.RACK_HEIGHT_THRESHOLD);
  const isSwingZone = wrist.y >= (shoulder.y + CONFIG.DETECTION.SWING_HEIGHT_THRESHOLD);

  // Classify
  if (isOverhead) {
    movementType = 'SNATCH';
  } else if (isRackZone) {
    // Check horizontal proximity to torso (clean has bell close to body)
    const torsoCenter = (state.lastPose[CONFIG.LEFT.SHOULDER].x + state.lastPose[CONFIG.RIGHT.SHOULDER].x) / 2;
    const horizontalDist = Math.abs(wrist.x - torsoCenter);

    if (horizontalDist < CONFIG.DETECTION.RACK_HORIZONTAL_PROXIMITY) {
      movementType = 'CLEAN';
    } else {
      movementType = 'SWING'; // Swing finishing at shoulder height
    }
  } else if (isSwingZone) {
    movementType = 'SWING';
  }

  // Build full classification
  if (config === 'SINGLE_BELL') {
    return `${movementType}_SINGLE_${hands.toUpperCase()}`;
  } else if (config === 'TWO_HANDS_ONE_BELL') {
    return `${movementType}_TWO_HANDS`;
  } else if (config === 'DOUBLE_BELLS') {
    return `${movementType}_DOUBLE`;
  }

  return `${movementType}_UNKNOWN_CONFIG`;
}

// ============================================
// REP DETECTION (Enhanced with Movement Classification)
// ============================================

function runSnatchLogic(pose) {
  const v = state.smoothedVelocity;
  const vy = state.smoothedVy;
  const idx = state.lockedSide === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const hip = pose[idx.HIP];
  const shoulder = pose[idx.SHOULDER];

  if (!wrist || !hip || !shoulder) return;

  // Logic Boundaries
  const isBelowHip = wrist.y > (hip.y - 0.05);
  const isAboveShoulder = wrist.y < shoulder.y;

  // --- STATE MACHINE ---

  // A. RESET: If we are in LOCKOUT or IDLE, we must drop below hip to start next rep.
  if (state.phase === "IDLE" || state.phase === "LOCKOUT") {
    if (isBelowHip) {
      state.phase = "BOTTOM";
      state.overheadHoldCount = 0;
      if (CONFIG.DEBUG_MODE) console.log("Phase: BOTTOM (Ready to Pull)");
    }
  }

  // B. PULL: Waiting for the upward movement to start concentric phase.
  else if (state.phase === "BOTTOM") {
    if (!isBelowHip && vy < -0.4) {
      state.phase = "CONCENTRIC";
      state.currentRepPeak = 0;
      if (CONFIG.DEBUG_MODE) console.log("Phase: CONCENTRIC (Pulling)");
    }
  }

  // C. POWER: Tracking peak speed and looking for the catch.
  else if (state.phase === "CONCENTRIC") {
    // 1. Capture absolute peak velocity during the flight
    if (v > state.currentRepPeak) state.currentRepPeak = v;

    // 2. Lockout Check (Stability above shoulder)
    const isStable = Math.abs(vy) < CONFIG.LOCKOUT_VY_CUTOFF && v < CONFIG.LOCKOUT_SPEED_CUTOFF;

    if (isAboveShoulder && isStable) {
      state.overheadHoldCount++;

      // Require 2 frames of stability
      if (state.overheadHoldCount >= 2) {
        recordRep();
        if (CONFIG.DEBUG_MODE) console.log(`📊 Rep ${state.repHistory.length} Recorded!`);
      }
    } else {
      // If they wobble out of stable zone, reset the hold timer
      state.overheadHoldCount = 0;

      // Safety: If they drop back down without locking out, reset to bottom
      if (isBelowHip) {
        state.phase = "BOTTOM";
        if (CONFIG.DEBUG_MODE) console.log("Phase: BOTTOM (Aborted/Dropped)");
      }
    }
  }
}

function recordRep() {
  state.phase = "LOCKOUT";
  state.overheadHoldCount = 0;

  // ✅ NEW: Detect movement type at lockout
  const movementType = classifyMovementAtLockout();

  if (state.session.currentSet) {
    // ✅ MODIFIED: Store rep with movement classification
    state.session.currentSet.reps.push({
      velocity: state.currentRepPeak,
      movement: movementType
    });
  }

  state.repHistory.push(state.currentRepPeak);

  // ✅ NEW: Update movement history
  state.movementDetection.movementHistory.push(movementType);
  if (state.movementDetection.movementHistory.length > 3) {
    state.movementDetection.movementHistory.shift();
  }

  // Calculate baseline (average of first 3 reps)
  if (state.repHistory.length === CONFIG.BASELINE_REPS) {
    state.baseline = state.repHistory.reduce((a,b) => a+b, 0) / CONFIG.BASELINE_REPS;
  }

  // Calculate drop-off percentage
  let dropPct = "--";
  let dropColor = "#fff";

  if (state.baseline > 0 && state.repHistory.length > CONFIG.BASELINE_REPS) {
    const drop = ((state.baseline - state.currentRepPeak) / state.baseline) * 100;
    dropPct = drop.toFixed(1) + "%";

    // Color coding
    if (drop < CONFIG.DROP_WARN) {
      dropColor = "#10b981"; // Green - good
    } else if (drop < CONFIG.DROP_FAIL) {
      dropColor = "#fbbf24"; // Yellow - warning
    } else {
      dropColor = "#ef4444"; // Red - fatigue
    }
  }

  updateUIValues(state.repHistory.length, state.currentRepPeak, dropPct, dropColor);

  // ✅ NEW: Update movement detection UI
  updateMovementDisplay(movementType, state.movementDetection.detectedConfig, state.movementDetection.detectedHands);

  if (CONFIG.DEBUG_MODE) {
    console.log(`📊 REP #${state.repHistory.length}: ${state.currentRepPeak.toFixed(2)} m/s | Drop: ${dropPct} | Movement: ${movementType}`);
  }
}

// ============================================
// HELPERS
// ============================================

function isWristInFloorZone(pose, side) {
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const knee = pose[idx.KNEE];
  if (!wrist || !knee) return false;
  return wrist.y > knee.y;
}

function resetSession() {
  state.session = { currentSet: null, history: [] };
  state.repHistory = [];
  state.baseline = 0;
  state.testStage = "IDLE";
  state.lockedSide = "unknown";
  state.activeTrackingSide = "left";
  state.armingSide = null;
  state.smoothedVelocity = 0;
  state.smoothedVy = 0;
  state.lastSpeed = 0;
  state.lastVy = 0;
  state.lockedCalibration = null;
  state.prevWrist = null;

  // ✅ NEW: Reset movement detection
  state.movementDetection = {
    leftWristVelocity: 0,
    rightWristVelocity: 0,
    leftPrevWrist: null,
    rightPrevWrist: null,
    wristSpacing: 0,
    detectedMovement: 'UNKNOWN',
    detectedConfig: 'UNKNOWN',
    detectedHands: 'none',
    movementHistory: []
  };

  updateUIValues(0, 0, "--", "#fff");
  setStatus("Session Cleared — Ready", "#3b82f6");
}

// ============================================
// UI (Enhanced with Movement Display)
// ============================================

function updateUIValues(reps, peak, drop = "--", dropColor = "#fff") {
  document.getElementById("val-reps").textContent = reps;
  document.getElementById("val-peak").textContent = peak.toFixed(2);

  const dropEl = document.getElementById("val-drop");
  if (dropEl) {
    dropEl.textContent = drop;
    dropEl.style.color = dropColor;
  }
}

function setStatus(text, color) {
  const pill = document.getElementById("status-pill");
  if (pill) {
    pill.textContent = text;
    pill.style.color = color;
    pill.style.borderColor = color;
  }
}

// ✅ NEW: Movement Display Functions

function updateMovementDisplay(movementType, configuration, hands) {
  const movementEl = document.getElementById('detected-movement');
  const configEl = document.getElementById('detected-config');
  const handsEl = document.getElementById('active-hands');
  const statusIndicator = document.getElementById('detection-status');

  if (!movementEl || !configEl || !handsEl || !statusIndicator) return;

  // Format movement name for display
  const displayName = formatMovementName(movementType);
  movementEl.textContent = displayName;

  // Add color class based on movement type
  movementEl.className = 'movement-text';
  if (movementType.includes('SWING')) {
    movementEl.classList.add('swing');
  } else if (movementType.includes('CLEAN')) {
    movementEl.classList.add('clean');
  } else if (movementType.includes('SNATCH')) {
    movementEl.classList.add('snatch');
  }

  // Update configuration
  configEl.textContent = formatConfiguration(configuration);

  // Update active hands
  handsEl.textContent = formatHands(hands);

  // Update status indicator
  statusIndicator.className = 'status-indicator locked';

  // Update movement history
  updateMovementHistory(movementType);
}

function formatMovementName(movementType) {
  const typeMap = {
    'SWING_SINGLE_LEFT': 'One Arm Swing (L)',
    'SWING_SINGLE_RIGHT': 'One Arm Swing (R)',
    'SWING_TWO_HANDS': 'Two Arm Swing',
    'SWING_DOUBLE': 'Double KB Swing',
    'CLEAN_SINGLE_LEFT': 'One Arm Clean (L)',
    'CLEAN_SINGLE_RIGHT': 'One Arm Clean (R)',
    'CLEAN_TWO_HANDS': 'Two Arm Clean',
    'CLEAN_DOUBLE': 'Double KB Clean',
    'SNATCH_SINGLE_LEFT': 'One Arm Snatch (L)',
    'SNATCH_SINGLE_RIGHT': 'One Arm Snatch (R)',
    'SNATCH_TWO_HANDS': 'Two Arm Snatch',
    'SNATCH_DOUBLE': 'Double KB Snatch'
  };
  return typeMap[movementType] || movementType.replace(/_/g, ' ');
}

function formatConfiguration(config) {
  const configMap = {
    'SINGLE_BELL': 'Single Kettlebell',
    'TWO_HANDS_ONE_BELL': 'Two Hands (1 KB)',
    'DOUBLE_BELLS': 'Double Kettlebells'
  };
  return configMap[config] || '—';
}

function formatHands(hands) {
  if (hands === 'both') return 'Both';
  if (hands === 'left') return 'Left';
  if (hands === 'right') return 'Right';
  return '—';
}

function updateMovementHistory(movementType) {
  const historyContainer = document.getElementById('history-badges');
  if (!historyContainer) return;

  // Add new badge
  const badge = document.createElement('span');
  badge.className = 'history-badge';
  badge.textContent = formatMovementName(movementType).replace(/\s*\(.\)/, '');

  historyContainer.insertBefore(badge, historyContainer.firstChild);

  // Keep only last 3
  while (historyContainer.children.length > 3) {
    historyContainer.removeChild(historyContainer.lastChild);
  }
}

function resetMovementDisplay() {
  const movementEl = document.getElementById('detected-movement');
  const configEl = document.getElementById('detected-config');
  const handsEl = document.getElementById('active-hands');
  const statusIndicator = document.getElementById('detection-status');
  const historyContainer = document.getElementById('history-badges');

  if (movementEl) {
    movementEl.textContent = 'Waiting...';
    movementEl.className = 'movement-text';
  }
  if (configEl) configEl.textContent = '—';
  if (handsEl) handsEl.textContent = '—';
  if (statusIndicator) statusIndicator.className = 'status-indicator detecting';
  if (historyContainer) historyContainer.innerHTML = '';
}

function drawOverlay() {
  if (!state.lastPose) return;

  // ✅ DEBUG OVERLAY (Top-left corner)
  if (CONFIG.DEBUG_MODE) {
    state.ctx.fillStyle = "#fbbf24";
    state.ctx.font = "12px monospace";
    state.ctx.fillText(`Side: ${state.testStage === "IDLE" ? state.activeTrackingSide : state.lockedSide}`, 10, 20);
    state.ctx.fillText(`Speed: ${state.lastSpeed.toFixed(2)} m/s`, 10, 35);
    state.ctx.fillText(`Vy: ${state.lastVy.toFixed(2)} m/s`, 10, 50);
    state.ctx.fillText(`Stage: ${state.testStage}`, 10, 65);
    state.ctx.fillText(`Parked: ${state.parkingConfirmed}`, 10, 80);

    // ✅ NEW: Movement detection debug info
    state.ctx.fillText(`L-Vel: ${state.movementDetection.leftWristVelocity.toFixed(2)} m/s`, 10, 95);
    state.ctx.fillText(`R-Vel: ${state.movementDetection.rightWristVelocity.toFixed(2)} m/s`, 10, 110);
    state.ctx.fillText(`Config: ${state.movementDetection.detectedConfig}`, 10, 125);
  }

  const side = state.testStage === "IDLE" ? state.activeTrackingSide : state.lockedSide;

  if (state.testStage === "RUNNING" && side !== "unknown") {
    const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
    const wrist = state.lastPose[idx.WRIST];
    drawDot(wrist, true, "#10b981");

    const inZone = isWristInFloorZone(state.lastPose, side);
    drawParkingLine(state.lastPose, side, inZone);
  } else {
    const lWrist = state.lastPose[CONFIG.LEFT.WRIST];
    const rWrist = state.lastPose[CONFIG.RIGHT.WRIST];
    const lY = lWrist?.y || 0;
    const rY = rWrist?.y || 0;
    const lowest = lY > rY ? "left" : "right";
    const color = "#fbbf24";

    drawDot(lWrist, lowest==="left", color);
    drawDot(rWrist, lowest==="right", color);
    drawParkingLine(state.lastPose, lowest, isWristInFloorZone(state.lastPose, lowest));
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

// ============================================
// EXPORT (Enhanced with Movement Data)
// ============================================

async function exportToMake() {
  const history = state.session.history;
  if (!history.length) {
    alert("No completed sets to export.");
    return;
  }

  const totalReps = history.reduce((sum, set) => sum + set.reps.length, 0);

  // ✅ BACKWARD COMPATIBLE: Calculate total average
  const allVelocities = history.flatMap(set => 
    set.reps.map(rep => typeof rep === 'number' ? rep : rep.velocity)
  );
  const totalAvg = allVelocities.length > 0 ?
    (allVelocities.reduce((a,b) => a+b, 0) / allVelocities.length) : 0;

  const payload = {
    athlete_id: "dad_ready_user",
    session_date: new Date().toISOString(),
    total_reps: totalReps,
    session_avg_velocity: totalAvg.toFixed(2),
    sets: history.map((set, index) => {
      // ✅ BACKWARD COMPATIBLE: Handle both number and object formats
      const rawPeaks = set.reps.map(rep => typeof rep === 'number' ? rep : rep.velocity);
      const movements = set.reps.map(rep => 
        typeof rep === 'object' ? rep.movement : `SNATCH_SINGLE_${set.hand.toUpperCase()}`
      );

      return {
        set_order: index + 1,
        hand: set.hand,
        rep_count: set.reps.length,
        peak_velocity_avg: parseFloat(set.avgVelocity),
        raw_peaks: rawPeaks,
        movements: movements // ✅ NEW FIELD
      };
    })
  };

  console.log("EXPORTING TO MAKE:", JSON.stringify(payload, null, 2));
  setStatus("Exporting to Make...", "#8b5cf6");

  try {
    const response = await fetch(CONFIG.MAKE_WEBHOOK_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if(response.ok) {
      setStatus("Success! Session Saved.", "#10b981");
    } else {
      setStatus("Error: Make.com rejected payload.", "#ef4444");
    }
  } catch(e) {
    console.error("Network Error:", e);
    setStatus("Network Error (Check Console)", "#ef4444");
  }
}

initializeApp();
