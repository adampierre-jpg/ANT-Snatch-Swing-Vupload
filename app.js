/**
 * VBT v3.7 - POSE-BASED MOVEMENT DETECTION
 * New logic: HINGE and RACK poses determine movement classification
 */

import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

// ============================================
// CONFIG
// ============================================

const CONFIG = {
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

  SMOOTHING_ALPHA: 0.15,
  MAX_REALISTIC_VELOCITY: 8.0,
  ZERO_BAND: 0.1,
  MIN_DT: 0.016,
  MAX_DT: 0.1,

  LOCKOUT_VY_CUTOFF: 0.6,
  LOCKOUT_SPEED_CUTOFF: 2.0,

  HINGE_GRACE_MS: 5000,
  RACK_LOCK_FRAMES: 30,  // ~1 second at 30fps

  BASELINE_REPS: 3,
  DROP_WARN: 15,
  DROP_FAIL: 20,

  MIN_DET_CONF: 0.5,
  MIN_TRACK_CONF: 0.5,

  MOVEMENT: {
    HINGE_THRESHOLD: 0.08,           // Shoulder-hip distance when hinged
    KNEE_CROSS_THRESHOLD: 0.05,      // Wrist must be this much below knee

    RACK_HEIGHT_MIN: -0.1,           // Rack position relative to shoulder
    RACK_HEIGHT_MAX: 0.15,
    RACK_HORIZONTAL_PROXIMITY: 0.18,

    OVERHEAD_MIN_HEIGHT: 0.05,       // Above shoulder for overhead

    SWING_MIN_ABOVE_HIP: 0.1,        // Swing must be above hip
    SWING_MAX_ABOVE_SHOULDER: 0.05,  // But below shoulder

    ROW_AT_HIP_TOLERANCE: 0.1,       // Row ends at hip level

    HIP_CROSS_TOLERANCE: 0.05,       // For detecting hip crossing

    BALLISTIC_VELOCITY: 2.5          // Press vs ballistic threshold
  },

  MAKE_WEBHOOK_URL: "https://hook.us2.make.com/0l88dnosrk2t8a29yfk83fej8hp8j3jk",

  DEBUG_MODE: true
};

// ============================================
// STATE
// ============================================

let state = {
  video: null,
  canvas: null,
  ctx: null,
  landmarker: null,
  isModelLoaded: false,
  isVideoReady: false,
  isTestRunning: false,

  timeMs: 0,
  lastPose: null,

  // Pose state
  currentPose: "NONE",              // HINGE, RACK, OVERHEAD, NONE
  poseLockedAt: 0,
  poseFrameCount: 0,

  // Side detection
  lockedSide: "unknown",
  activeTrackingSide: "left",

  // Physics
  prevWrist: null,
  lockedCalibration: null,
  smoothedVelocity: 0,
  smoothedVy: 0,
  lastSpeed: 0,
  lastVy: 0,

  // Movement tracking
  phase: "IDLE",                    // IDLE, PULLING, LOCKED
  movementStartPose: null,          // HINGE or RACK
  currentRepPeak: 0,
  peakWristY: 1.0,

  // Hip crossing
  wristAboveHip: false,
  hipCrossedUpward: false,

  // Session
  session: {
    currentSet: null,
    history: []
  },

  // History arrays
  cleanHistory: [],
  pressHistory: [],
  snatchHistory: [],
  swingHistory: [],
  rowHistory: [],

  // Baselines
  cleanBaseline: 0,
  pressBaseline: 0,
  snatchBaseline: 0,
  swingBaseline: 0,
  rowBaseline: 0
};

// ============================================
// INITIALIZATION (unchanged from v3.6)
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
    document.getElementById("btn-start-test").textContent = "Pause Test";
    document.getElementById("btn-reset").disabled = false;
    setStatus("Scanning for HINGE pose...", "#fbbf24");
    resetMovementDisplay();

    if (state.video.paused) {
      try { await state.video.play(); } catch(e) {}
    }
  } else {
    state.isTestRunning = false;
    document.getElementById("btn-start-test").textContent = "Resume Test";
    setStatus("Paused", "#fbbf24");
    state.video.pause();
  }
}

// ============================================
// MASTER LOOP (unchanged)
// ============================================

async function masterLoop(timestamp) {
  requestAnimationFrame(masterLoop);
  if (!state.isModelLoaded || !state.video) return;

  state.timeMs = timestamp;

  state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
  state.ctx.drawImage(state.video, 0, 0, state.canvas.width, state.canvas.height);

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

  if (state.isTestRunning) {
    runPhysics(pose, state.timeMs);
    updatePoseState(pose, state.timeMs);
    runMovementLogic(pose, state.timeMs);
  }

  drawOverlay();
}

// ============================================
// PHYSICS ENGINE (unchanged from v3.6)
// ============================================

function runPhysics(pose, timeMs) {
  const side = state.lockedSide === "unknown" ? state.activeTrackingSide : state.lockedSide;
  if (!side || side === "unknown") return;

  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const shoulder = pose[idx.SHOULDER];
  const hip = pose[idx.HIP];
  if (!wrist || !shoulder || !hip) return;

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

  const TARGET_FPS = 30;
  const frameTimeMs = 1000 / TARGET_FPS;
  const actualFrameTimeMs = timeMs - state.prevWrist.t;
  const timeRatio = frameTimeMs / actualFrameTimeMs;
  vx *= timeRatio;
  vy *= timeRatio;
  speed = Math.hypot(vx, vy);

  if (speed < CONFIG.ZERO_BAND) speed = 0;

  state.smoothedVelocity = CONFIG.SMOOTHING_ALPHA * speed + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVelocity;
  state.smoothedVy = CONFIG.SMOOTHING_ALPHA * vy + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVy;

  state.lastSpeed = Math.min(state.smoothedVelocity, CONFIG.MAX_REALISTIC_VELOCITY);
  state.lastVy = Math.min(Math.max(state.smoothedVy, -CONFIG.MAX_REALISTIC_VELOCITY), CONFIG.MAX_REALISTIC_VELOCITY);

  state.prevWrist = { x: wrist.x, y: wrist.y, t: timeMs };

  if (state.lockedSide !== "unknown") {
    document.getElementById("val-velocity").textContent = state.lastSpeed.toFixed(2);
  }
}

// ============================================
// POSE DETECTION HELPERS
// ============================================

function isHinged(pose, side) {
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const shoulder = pose[idx.SHOULDER];
  const hip = pose[idx.HIP];

  if (!shoulder || !hip) return false;

  // Hinged when shoulder-hip vertical distance is small
  const verticalDist = Math.abs(shoulder.y - hip.y);
  return verticalDist < CONFIG.MOVEMENT.HINGE_THRESHOLD;
}

function isWristBelowKnee(pose, side) {
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const knee = pose[idx.KNEE];

  if (!wrist || !knee) return false;

  return wrist.y > (knee.y + CONFIG.MOVEMENT.KNEE_CROSS_THRESHOLD);
}

function isWristAtRack(pose, side) {
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const shoulder = pose[idx.SHOULDER];

  if (!wrist || !shoulder) return false;

  const lShoulder = pose[CONFIG.LEFT.SHOULDER];
  const rShoulder = pose[CONFIG.RIGHT.SHOULDER];
  const torsoCenter = (lShoulder.x + rShoulder.x) / 2;
  const horizontalDist = Math.abs(wrist.x - torsoCenter);

  const atShoulderHeight = wrist.y >= (shoulder.y + CONFIG.MOVEMENT.RACK_HEIGHT_MIN) && 
                           wrist.y <= (shoulder.y + CONFIG.MOVEMENT.RACK_HEIGHT_MAX);
  const nearMidline = horizontalDist < CONFIG.MOVEMENT.RACK_HORIZONTAL_PROXIMITY;

  return atShoulderHeight && nearMidline;
}

function getWristHeightZone(pose, side) {
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const shoulder = pose[idx.SHOULDER];
  const hip = pose[idx.HIP];

  if (!wrist || !shoulder || !hip) return 'UNKNOWN';

  // Overhead
  if (wrist.y < (shoulder.y - CONFIG.MOVEMENT.OVERHEAD_MIN_HEIGHT)) {
    return 'OVERHEAD';
  }

  // At shoulder (rack area)
  if (wrist.y >= (shoulder.y - CONFIG.MOVEMENT.OVERHEAD_MIN_HEIGHT) && 
      wrist.y <= (shoulder.y + CONFIG.MOVEMENT.RACK_HEIGHT_MAX)) {
    return 'SHOULDER';
  }

  // Between shoulder and hip (swing zone)
  if (wrist.y > (shoulder.y + CONFIG.MOVEMENT.RACK_HEIGHT_MAX) && wrist.y < hip.y) {
    return 'MID';
  }

  // At hip
  if (Math.abs(wrist.y - hip.y) < CONFIG.MOVEMENT.ROW_AT_HIP_TOLERANCE) {
    return 'HIP';
  }

  // Below hip
  return 'LOW';
}

function isWristNearMidline(pose, side) {
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];

  const lShoulder = pose[CONFIG.LEFT.SHOULDER];
  const rShoulder = pose[CONFIG.RIGHT.SHOULDER];
  const torsoCenter = (lShoulder.x + rShoulder.x) / 2;
  const horizontalDist = Math.abs(wrist.x - torsoCenter);

  return horizontalDist < CONFIG.MOVEMENT.RACK_HORIZONTAL_PROXIMITY;
}

// ============================================
// POSE STATE MACHINE
// ============================================

function updatePoseState(pose, timeMs) {
  const side = state.lockedSide === "unknown" ? state.activeTrackingSide : state.lockedSide;

  // Determine active side if not locked
  if (state.lockedSide === "unknown") {
    const lWrist = pose[CONFIG.LEFT.WRIST];
    const rWrist = pose[CONFIG.RIGHT.WRIST];
    if (lWrist && rWrist) {
      state.activeTrackingSide = lWrist.y > rWrist.y ? "left" : "right";
    }
  }

  // Check for HINGE pose (set start)
  if (state.currentPose === "NONE" && isWristBelowKnee(pose, side)) {
    state.currentPose = "HINGE";
    state.poseLockedAt = timeMs;
    state.lockedSide = side;

    if (!state.session.currentSet) {
      startNewSet(side, timeMs);
    }

    if (CONFIG.DEBUG_MODE) console.log(`✓ HINGE locked [${side}]`);
    setStatus(`HINGE locked [${side.toUpperCase()}]`, "#10b981");
  }

  // Check for RACK pose (needs sustained hold)
  if (isWristAtRack(pose, side) && !isHinged(pose, side)) {
    if (state.currentPose !== "RACK") {
      state.poseFrameCount = 0;
    }
    state.poseFrameCount++;

    if (state.poseFrameCount >= CONFIG.RACK_LOCK_FRAMES && state.currentPose !== "RACK") {
      state.currentPose = "RACK";
      state.poseLockedAt = timeMs;
      if (CONFIG.DEBUG_MODE) console.log(`✓ RACK locked`);
    }
  } else {
    if (state.currentPose === "RACK") {
      // Left rack
      state.currentPose = "NONE";
      state.poseFrameCount = 0;
    }
  }

  // Check for set ending (standing from hinge with grace period)
  if (state.session.currentSet) {
    const graceExpired = (timeMs - state.poseLockedAt) > CONFIG.HINGE_GRACE_MS;

    if (graceExpired && !isHinged(pose, side) && !isWristBelowKnee(pose, side) && state.lastVy < -0.3) {
      // Standing up after grace period
      if (CONFIG.DEBUG_MODE) console.log(`🛑 Set ended - standing detected`);
      endCurrentSet();
    }
  }
}

// ============================================
// MOVEMENT DETECTION LOGIC - NEW POSE-BASED
// ============================================

function runMovementLogic(pose, timeMs) {
  if (!state.session.currentSet) return;

  const side = state.lockedSide;
  if (side === "unknown") return;

  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const hip = pose[idx.HIP];
  if (!wrist || !hip) return;

  const v = state.smoothedVelocity;
  const vy = state.smoothedVy;
  const isStable = Math.abs(vy) < CONFIG.LOCKOUT_VY_CUTOFF && v < CONFIG.LOCKOUT_SPEED_CUTOFF;
  const heightZone = getWristHeightZone(pose, side);
  const hinged = isHinged(pose, side);
  const nearMidline = isWristNearMidline(pose, side);

  // Track hip crossing
  const currentlyAboveHip = wrist.y < hip.y;
  if (currentlyAboveHip && !state.wristAboveHip) {
    state.hipCrossedUpward = true;
  }
  state.wristAboveHip = currentlyAboveHip;

  // IDLE - waiting for movement to start
  if (state.phase === "IDLE") {
    if (state.currentPose === "HINGE" && vy < -0.4) {
      state.phase = "PULLING";
      state.movementStartPose = "HINGE";
      state.currentRepPeak = 0;
      state.peakWristY = wrist.y;
      state.hipCrossedUpward = false;
      state.wristAboveHip = wrist.y < hip.y;

      if (CONFIG.DEBUG_MODE) console.log(`→ PULLING from HINGE`);
    }
    else if (state.currentPose === "RACK" && Math.abs(vy) > 0.4) {
      state.phase = "PULLING";
      state.movementStartPose = "RACK";
      state.currentRepPeak = 0;
      state.peakWristY = wrist.y;
      state.hipCrossedUpward = false;
      state.wristAboveHip = wrist.y < hip.y;

      if (CONFIG.DEBUG_MODE) console.log(`→ PULLING from RACK`);
    }
  }

  // PULLING - track peak velocity and detect lockout
  else if (state.phase === "PULLING") {
    if (v > state.currentRepPeak) state.currentRepPeak = v;
    if (wrist.y < state.peakWristY) state.peakWristY = wrist.y;

    // Check for lockout at different heights
    if (isStable) {

      // From HINGE - can be Clean, Swing, Snatch, or Row
      if (state.movementStartPose === "HINGE") {

        // OVERHEAD = Snatch
        if (heightZone === 'OVERHEAD') {
          recordSnatch(pose);
          state.phase = "LOCKED";
        }

        // SHOULDER + midline + standing = Clean
        else if (heightZone === 'SHOULDER' && nearMidline && !hinged && state.hipCrossedUpward) {
          recordClean(pose, "CLEAN_FROM_FLOOR");
          state.phase = "LOCKED";
        }

        // MID (between hip and shoulder) = Swing
        else if (heightZone === 'MID' && state.hipCrossedUpward) {
          recordSwing(pose);
          state.phase = "LOCKED";
        }

        // HIP + hinged + no hip cross = Row
        else if (heightZone === 'HIP' && hinged && !state.hipCrossedUpward) {
          recordRow(pose);
          state.phase = "LOCKED";
        }
      }

      // From RACK - can be Press or Re-clean
      else if (state.movementStartPose === "RACK") {

        // OVERHEAD = Press
        if (heightZone === 'OVERHEAD') {
          recordPress(pose);
          state.phase = "LOCKED";
        }

        // Back to SHOULDER after dip = Re-clean
        else if (heightZone === 'SHOULDER' && nearMidline && state.hipCrossedUpward) {
          recordClean(pose, "RE_CLEAN");
          state.phase = "LOCKED";
        }
      }
    }
  }

  // LOCKED - movement complete, wait for next
  else if (state.phase === "LOCKED") {
    // Reset to idle when returning to start position
    if (isWristBelowKnee(pose, side) || (state.currentPose === "RACK")) {
      state.phase = "IDLE";
      state.movementStartPose = null;
      if (CONFIG.DEBUG_MODE) console.log(`← IDLE (ready for next)`);
    }
  }
}

// ============================================
// RECORDING FUNCTIONS
// ============================================

function recordClean(pose, cleanType) {
  const cleanData = {
    type: cleanType,
    velocity: state.currentRepPeak,
    timestamp: Date.now()
  };

  if (state.session.currentSet) {
    state.session.currentSet.cleans.push(cleanData);
  }

  state.cleanHistory.push(state.currentRepPeak);

  if (state.cleanHistory.length === CONFIG.BASELINE_REPS && !state.cleanBaseline) {
    state.cleanBaseline = state.cleanHistory.reduce((a,b) => a+b, 0) / CONFIG.BASELINE_REPS;
  }

  let dropPct = "--";
  let dropColor = "#fff";
  if (state.cleanBaseline > 0 && state.cleanHistory.length > CONFIG.BASELINE_REPS) {
    const drop = ((state.cleanBaseline - state.currentRepPeak) / state.cleanBaseline) * 100;
    dropPct = drop.toFixed(1) + "%";

    if (drop < CONFIG.DROP_WARN) {
      dropColor = "#10b981";
    } else if (drop < CONFIG.DROP_FAIL) {
      dropColor = "#fbbf24";
    } else {
      dropColor = "#ef4444";
    }
  }

  const displayType = cleanType === "CLEAN_FROM_FLOOR" ? "Clean (Floor)" : "Re-Clean";

  if (CONFIG.DEBUG_MODE) {
    console.log(`✅ ${displayType} #${state.cleanHistory.length}: ${state.currentRepPeak.toFixed(2)} m/s | Drop: ${dropPct}`);
  }

  updateCleanDisplay(state.cleanHistory.length, state.currentRepPeak, dropPct, dropColor);
  updateMovementDisplay(cleanType);
  updateTotalReps();
}

function recordPress(pose) {
  const pressData = {
    type: 'PRESS',
    velocity: state.currentRepPeak,
    timestamp: Date.now()
  };

  if (state.session.currentSet) {
    state.session.currentSet.presses.push(pressData);
  }

  state.pressHistory.push(state.currentRepPeak);

  if (state.pressHistory.length === CONFIG.BASELINE_REPS && !state.pressBaseline) {
    state.pressBaseline = state.pressHistory.reduce((a,b) => a+b, 0) / CONFIG.BASELINE_REPS;
  }

  let dropPct = "--";
  let dropColor = "#fff";
  if (state.pressBaseline > 0 && state.pressHistory.length > CONFIG.BASELINE_REPS) {
    const drop = ((state.pressBaseline - state.currentRepPeak) / state.pressBaseline) * 100;
    dropPct = drop.toFixed(1) + "%";

    if (drop < CONFIG.DROP_WARN) {
      dropColor = "#10b981";
    } else if (drop < CONFIG.DROP_FAIL) {
      dropColor = "#fbbf24";
    } else {
      dropColor = "#ef4444";
    }
  }

  if (CONFIG.DEBUG_MODE) {
    console.log(`💪 PRESS #${state.pressHistory.length}: ${state.currentRepPeak.toFixed(2)} m/s | Drop: ${dropPct}`);
  }

  updatePressDisplay(state.pressHistory.length, state.currentRepPeak, dropPct, dropColor);
  updateMovementDisplay(`PRESS_SINGLE_${state.lockedSide.toUpperCase()}`);
  updateTotalReps();
}

function recordSnatch(pose) {
  const snatchData = {
    type: 'SNATCH',
    velocity: state.currentRepPeak,
    timestamp: Date.now()
  };

  if (state.session.currentSet) {
    state.session.currentSet.snatches.push(snatchData);
  }

  state.snatchHistory.push(state.currentRepPeak);

  if (state.snatchHistory.length === CONFIG.BASELINE_REPS && !state.snatchBaseline) {
    state.snatchBaseline = state.snatchHistory.reduce((a,b) => a+b, 0) / CONFIG.BASELINE_REPS;
  }

  let dropPct = "--";
  let dropColor = "#fff";
  if (state.snatchBaseline > 0 && state.snatchHistory.length > CONFIG.BASELINE_REPS) {
    const drop = ((state.snatchBaseline - state.currentRepPeak) / state.snatchBaseline) * 100;
    dropPct = drop.toFixed(1) + "%";

    if (drop < CONFIG.DROP_WARN) {
      dropColor = "#10b981";
    } else if (drop < CONFIG.DROP_FAIL) {
      dropColor = "#fbbf24";
    } else {
      dropColor = "#ef4444";
    }
  }

  if (CONFIG.DEBUG_MODE) {
    console.log(`⚡ SNATCH #${state.snatchHistory.length}: ${state.currentRepPeak.toFixed(2)} m/s | Drop: ${dropPct}`);
  }

  updateSnatchDisplay(state.snatchHistory.length, state.currentRepPeak, dropPct, dropColor);
  updateMovementDisplay(`SNATCH_SINGLE_${state.lockedSide.toUpperCase()}`);
  updateTotalReps();
}

function recordSwing(pose) {
  const swingData = {
    type: 'SWING',
    velocity: state.currentRepPeak,
    timestamp: Date.now()
  };

  if (state.session.currentSet) {
    state.session.currentSet.swings.push(swingData);
  }

  state.swingHistory.push(state.currentRepPeak);

  if (state.swingHistory.length === CONFIG.BASELINE_REPS && !state.swingBaseline) {
    state.swingBaseline = state.swingHistory.reduce((a,b) => a+b, 0) / CONFIG.BASELINE_REPS;
  }

  let dropPct = "--";
  let dropColor = "#fff";
  if (state.swingBaseline > 0 && state.swingHistory.length > CONFIG.BASELINE_REPS) {
    const drop = ((state.swingBaseline - state.currentRepPeak) / state.swingBaseline) * 100;
    dropPct = drop.toFixed(1) + "%";

    if (drop < CONFIG.DROP_WARN) {
      dropColor = "#10b981";
    } else if (drop < CONFIG.DROP_FAIL) {
      dropColor = "#fbbf24";
    } else {
      dropColor = "#ef4444";
    }
  }

  if (CONFIG.DEBUG_MODE) {
    console.log(`🔄 SWING #${state.swingHistory.length}: ${state.currentRepPeak.toFixed(2)} m/s | Drop: ${dropPct}`);
  }

  updateSwingDisplay(state.swingHistory.length, state.currentRepPeak, dropPct, dropColor);
  updateMovementDisplay(`SWING_SINGLE_${state.lockedSide.toUpperCase()}`);
  updateTotalReps();
}

function recordRow(pose) {
  const rowData = {
    type: 'ROW',
    velocity: state.currentRepPeak,
    timestamp: Date.now()
  };

  if (state.session.currentSet) {
    state.session.currentSet.rows.push(rowData);
  }

  state.rowHistory.push(state.currentRepPeak);

  if (state.rowHistory.length === CONFIG.BASELINE_REPS && !state.rowBaseline) {
    state.rowBaseline = state.rowHistory.reduce((a,b) => a+b, 0) / CONFIG.BASELINE_REPS;
  }

  let dropPct = "--";
  let dropColor = "#fff";
  if (state.rowBaseline > 0 && state.rowHistory.length > CONFIG.BASELINE_REPS) {
    const drop = ((state.rowBaseline - state.currentRepPeak) / state.rowBaseline) * 100;
    dropPct = drop.toFixed(1) + "%";

    if (drop < CONFIG.DROP_WARN) {
      dropColor = "#10b981";
    } else if (drop < CONFIG.DROP_FAIL) {
      dropColor = "#fbbf24";
    } else {
      dropColor = "#ef4444";
    }
  }

  if (CONFIG.DEBUG_MODE) {
    console.log(`🚣 ROW #${state.rowHistory.length}: ${state.currentRepPeak.toFixed(2)} m/s | Drop: ${dropPct}`);
  }

  updateRowDisplay(state.rowHistory.length, state.currentRepPeak, dropPct, dropColor);
  updateMovementDisplay(`ROW_SINGLE_${state.lockedSide.toUpperCase()}`);
  updateTotalReps();
}

// ============================================
// SET MANAGEMENT
// ============================================

function startNewSet(side, timeMs) {
  state.lockedSide = side;
  state.poseLockedAt = timeMs;

  state.cleanHistory = [];
  state.pressHistory = [];
  state.snatchHistory = [];
  state.swingHistory = [];
  state.rowHistory = [];

  state.cleanBaseline = 0;
  state.pressBaseline = 0;
  state.snatchBaseline = 0;
  state.swingBaseline = 0;
  state.rowBaseline = 0;

  state.currentRepPeak = 0;
  state.phase = "IDLE";
  state.movementStartPose = null;

  state.session.currentSet = {
    id: state.session.history.length + 1,
    hand: side,
    cleans: [],
    presses: [],
    snatches: [],
    swings: [],
    rows: [],
    startTime: new Date()
  };

  const countEls = ['val-cleans', 'val-presses', 'val-snatches', 'val-swings', 'val-rows', 'val-total-reps'];
  countEls.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = '0';
  });

  const velEls = ['val-clean-velocity', 'val-press-velocity', 'val-snatch-velocity', 'val-swing-velocity', 'val-row-velocity'];
  velEls.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = '0.00';
  });

  const dropEls = ['val-clean-drop', 'val-press-drop', 'val-snatch-drop', 'val-swing-drop', 'val-row-drop'];
  dropEls.forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.textContent = '--';
      el.style.color = '#fff';
    }
  });

  if (CONFIG.DEBUG_MODE) console.log(`🚀 Set Started [${side}]`);
}

function endCurrentSet() {
  if (state.session.currentSet) {
    state.session.currentSet.endTime = new Date();

    state.session.currentSet.summary = {
      total_cleans: state.session.currentSet.cleans.length,
      floor_cleans: state.session.currentSet.cleans.filter(c => c.type === 'CLEAN_FROM_FLOOR').length,
      re_cleans: state.session.currentSet.cleans.filter(c => c.type === 'RE_CLEAN').length,
      total_presses: state.session.currentSet.presses.length,
      total_snatches: state.session.currentSet.snatches.length,
      total_swings: state.session.currentSet.swings.length,
      total_rows: state.session.currentSet.rows.length
    };

    state.session.history.push(state.session.currentSet);
  }

  state.lockedSide = "unknown";
  state.session.currentSet = null;
  state.activeTrackingSide = "left";
  state.currentPose = "NONE";
  state.phase = "IDLE";

  setStatus("Set Saved. Enter HINGE to start next.", "#3b82f6");
}

// ============================================
// UI UPDATES (unchanged from v3.6)
// ============================================

function updateCleanDisplay(count, velocity, drop, dropColor) {
  const countEl = document.getElementById("val-cleans");
  const velEl = document.getElementById("val-clean-velocity");
  const dropEl = document.getElementById("val-clean-drop");

  if (countEl) countEl.textContent = count;
  if (velEl) velEl.textContent = velocity.toFixed(2);
  if (dropEl) {
    dropEl.textContent = drop;
    dropEl.style.color = dropColor;
  }
}

function updatePressDisplay(count, velocity, drop, dropColor) {
  const countEl = document.getElementById("val-presses");
  const velEl = document.getElementById("val-press-velocity");
  const dropEl = document.getElementById("val-press-drop");

  if (countEl) countEl.textContent = count;
  if (velEl) velEl.textContent = velocity.toFixed(2);
  if (dropEl) {
    dropEl.textContent = drop;
    dropEl.style.color = dropColor;
  }
}

function updateSnatchDisplay(count, velocity, drop, dropColor) {
  const countEl = document.getElementById("val-snatches");
  const velEl = document.getElementById("val-snatch-velocity");
  const dropEl = document.getElementById("val-snatch-drop");

  if (countEl) countEl.textContent = count;
  if (velEl) velEl.textContent = velocity.toFixed(2);
  if (dropEl) {
    dropEl.textContent = drop;
    dropEl.style.color = dropColor;
  }
}

function updateSwingDisplay(count, velocity, drop, dropColor) {
  const countEl = document.getElementById("val-swings");
  const velEl = document.getElementById("val-swing-velocity");
  const dropEl = document.getElementById("val-swing-drop");

  if (countEl) countEl.textContent = count;
  if (velEl) velEl.textContent = velocity.toFixed(2);
  if (dropEl) {
    dropEl.textContent = drop;
    dropEl.style.color = dropColor;
  }
}

function updateRowDisplay(count, velocity, drop, dropColor) {
  const countEl = document.getElementById("val-rows");
  const velEl = document.getElementById("val-row-velocity");
  const dropEl = document.getElementById("val-row-drop");

  if (countEl) countEl.textContent = count;
  if (velEl) velEl.textContent = velocity.toFixed(2);
  if (dropEl) {
    dropEl.textContent = drop;
    dropEl.style.color = dropColor;
  }
}

function updateTotalReps() {
  if (!state.session.currentSet) return;

  const cleans = (state.session.currentSet.cleans || []).length;
  const presses = (state.session.currentSet.presses || []).length;
  const snatches = (state.session.currentSet.snatches || []).length;
  const swings = (state.session.currentSet.swings || []).length;
  const rows = (state.session.currentSet.rows || []).length;

  const total = cleans + presses + snatches + swings + rows;

  const totalEl = document.getElementById("val-total-reps");
  if (totalEl) totalEl.textContent = total;
}

function updateMovementDisplay(movementType) {
  const movementEl = document.getElementById('detected-movement');
  const configEl = document.getElementById('detected-config');
  const handsEl = document.getElementById('active-hands');
  const statusIndicator = document.getElementById('detection-status');

  if (!movementEl || !configEl || !handsEl || !statusIndicator) return;

  const displayName = formatMovementName(movementType);
  movementEl.textContent = displayName;

  movementEl.className = 'movement-text';
  if (movementType.includes('SWING')) {
    movementEl.classList.add('swing');
  } else if (movementType.includes('CLEAN')) {
    movementEl.classList.add('clean');
  } else if (movementType.includes('SNATCH')) {
    movementEl.classList.add('snatch');
  } else if (movementType.includes('PRESS')) {
    movementEl.classList.add('press');
  } else if (movementType.includes('ROW')) {
    movementEl.classList.add('press');
  }

  configEl.textContent = 'Single Kettlebell';
  handsEl.textContent = state.lockedSide === 'left' ? 'Left' : state.lockedSide === 'right' ? 'Right' : '—';
  statusIndicator.className = 'status-indicator locked';
}

function formatMovementName(movementType) {
  const typeMap = {
    'SWING_SINGLE_LEFT': 'One Arm Swing (L)',
    'SWING_SINGLE_RIGHT': 'One Arm Swing (R)',
    'CLEAN_FROM_FLOOR': 'Clean from Floor',
    'RE_CLEAN': 'Re-Clean',
    'SNATCH_SINGLE_LEFT': 'One Arm Snatch (L)',
    'SNATCH_SINGLE_RIGHT': 'One Arm Snatch (R)',
    'PRESS_SINGLE_LEFT': 'One Arm Press (L)',
    'PRESS_SINGLE_RIGHT': 'One Arm Press (R)',
    'ROW_SINGLE_LEFT': 'Sumo Row (L)',
    'ROW_SINGLE_RIGHT': 'Sumo Row (R)'
  };
  return typeMap[movementType] || movementType.replace(/_/g, ' ');
}

function resetMovementDisplay() {
  const movementEl = document.getElementById('detected-movement');
  const configEl = document.getElementById('detected-config');
  const handsEl = document.getElementById('active-hands');
  const statusIndicator = document.getElementById('detection-status');

  if (movementEl) {
    movementEl.textContent = 'Waiting...';
    movementEl.className = 'movement-text';
  }
  if (configEl) configEl.textContent = '—';
  if (handsEl) handsEl.textContent = '—';
  if (statusIndicator) statusIndicator.className = 'status-indicator detecting';
}

function setStatus(text, color) {
  const pill = document.getElementById("status-pill");
  if (pill) {
    pill.textContent = text;
    pill.style.color = color;
    pill.style.borderColor = color;
  }
}

// ============================================
// DRAWING (updated to show pose state)
// ============================================

function drawOverlay() {
  if (!state.lastPose) return;

  if (CONFIG.DEBUG_MODE) {
    state.ctx.fillStyle = "#fbbf24";
    state.ctx.font = "12px monospace";
    state.ctx.fillText(`Side: ${state.lockedSide}`, 10, 20);
    state.ctx.fillText(`Pose: ${state.currentPose}`, 10, 35);
    state.ctx.fillText(`Phase: ${state.phase}`, 10, 50);
    state.ctx.fillText(`Speed: ${state.lastSpeed.toFixed(2)} m/s`, 10, 65);
    state.ctx.fillText(`Vy: ${state.lastVy.toFixed(2)} m/s`, 10, 80);

    if (state.lockedSide !== "unknown") {
      const zone = getWristHeightZone(state.lastPose, state.lockedSide);
      const hinged = isHinged(state.lastPose, state.lockedSide);
      state.ctx.fillText(`Zone: ${zone}`, 10, 95);
      state.ctx.fillText(`Hinged: ${hinged}`, 10, 110);
      state.ctx.fillText(`Hip Cross: ${state.hipCrossedUpward}`, 10, 125);
    }
  }

  const side = state.lockedSide === "unknown" ? state.activeTrackingSide : state.lockedSide;

  if (state.lockedSide !== "unknown") {
    const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
    const wrist = state.lastPose[idx.WRIST];
    const knee = state.lastPose[idx.KNEE];

    // Draw wrist
    drawDot(wrist, true, "#10b981");

    // Draw knee line (HINGE detection line)
    if (knee) {
      const y = knee.y * state.canvas.height;
      state.ctx.beginPath();
      state.ctx.strokeStyle = state.currentPose === "HINGE" ? "#10b981" : "rgba(255,255,255,0.2)";
      state.ctx.lineWidth = 2;
      state.ctx.moveTo(0, y);
      state.ctx.lineTo(state.canvas.width, y);
      state.ctx.stroke();
    }
  } else {
    const lWrist = state.lastPose[CONFIG.LEFT.WRIST];
    const rWrist = state.lastPose[CONFIG.RIGHT.WRIST];
    const lY = lWrist?.y || 0;
    const rY = rWrist?.y || 0;
    const lowest = lY > rY ? "left" : "right";

    drawDot(lWrist, lowest==="left", "#fbbf24");
    drawDot(rWrist, lowest==="right", "#fbbf24");
  }
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
// SESSION MANAGEMENT
// ============================================

function resetSession() {
  state.session = { currentSet: null, history: [] };
  state.cleanHistory = [];
  state.pressHistory = [];
  state.snatchHistory = [];
  state.swingHistory = [];
  state.rowHistory = [];

  state.cleanBaseline = 0;
  state.pressBaseline = 0;
  state.snatchBaseline = 0;
  state.swingBaseline = 0;
  state.rowBaseline = 0;

  state.currentPose = "NONE";
  state.poseLockedAt = 0;
  state.poseFrameCount = 0;
  state.lockedSide = "unknown";
  state.activeTrackingSide = "left";
  state.smoothedVelocity = 0;
  state.smoothedVy = 0;
  state.lastSpeed = 0;
  state.lastVy = 0;
  state.lockedCalibration = null;
  state.prevWrist = null;
  state.phase = "IDLE";
  state.currentRepPeak = 0;
  state.peakWristY = 1.0;
  state.movementStartPose = null;
  state.wristAboveHip = false;
  state.hipCrossedUpward = false;

  if (state.video && state.video.src) {
    state.video.pause();
    state.video.currentTime = 0;
  }

  const countEls = ['val-cleans', 'val-presses', 'val-snatches', 'val-swings', 'val-rows', 'val-total-reps'];
  countEls.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = '0';
  });

  const velEls = ['val-clean-velocity', 'val-press-velocity', 'val-snatch-velocity', 'val-swing-velocity', 'val-row-velocity', 'val-velocity'];
  velEls.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = '0.00';
  });

  const dropEls = ['val-clean-drop', 'val-press-drop', 'val-snatch-drop', 'val-swing-drop', 'val-row-drop'];
  dropEls.forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.textContent = '--';
      el.style.color = '#fff';
    }
  });

  resetMovementDisplay();
  setStatus("Session Cleared — Ready", "#3b82f6");

  if (CONFIG.DEBUG_MODE) {
    console.log("🔄 Session Reset - All state cleared");
  }
}

// ============================================
// EXPORT
// ============================================

async function exportToMake() {
  const history = state.session.history;
  if (!history.length) {
    alert("No completed sets to export.");
    return;
  }

  const payload = {
    athlete_id: "dad_ready_user",
    session_date: new Date().toISOString(),
    sets: history.map((set, index) => ({
      set_order: index + 1,
      hand: set.hand,
      cleans: set.cleans || [],
      presses: set.presses || [],
      snatches: set.snatches || [],
      swings: set.swings || [],
      rows: set.rows || [],
      summary: set.summary || {}
    }))
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
