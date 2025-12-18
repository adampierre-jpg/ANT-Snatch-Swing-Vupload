/**
 * Version 3.4 CORRECTED - Movement Detection Using Existing State Machine
 * - Uses proven snatch detection logic as foundation
 * - Adds movement classification based on ACTUAL lockout criteria
 * - No parallel tracking - uses existing phase transitions
 */

import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

// ============================================
// CONFIG (Version 3.4 - Corrected)
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
  RESET_GRACE_MS_AFTER_LOCK: 5000,
  HIKE_VY_THRESHOLD: 0.3,
  HIKE_SPEED_THRESHOLD: 0.5,

  // REP & DROP-OFF LOGIC
  BASELINE_REPS: 3,
  DROP_WARN: 15,
  DROP_FAIL: 20,

  // MEDIAPIPE SETTINGS
  MIN_DET_CONF: 0.5,
  MIN_TRACK_CONF: 0.5,

  // ✅ MOVEMENT CLASSIFICATION CRITERIA (Based on Proven Logic)
  MOVEMENT: {
    // Snatch: Must lock out OVERHEAD (existing logic already checks this)
    SNATCH_MIN_HEIGHT_ABOVE_SHOULDER: 0.05,  // Wrist clearly above shoulder

    // Clean: Locks at RACK position (chest/shoulder height, close to torso)
    CLEAN_RACK_HEIGHT_MIN: -0.1,  // Below shoulder is OK
    CLEAN_RACK_HEIGHT_MAX: 0.15,  // Above shoulder is OK (rack range)
    CLEAN_HORIZONTAL_PROXIMITY: 0.18,  // Must be close to torso centerline

    // Swing: Locks at SHOULDER height or below (doesn't go overhead)
    SWING_MAX_HEIGHT_ABOVE_SHOULDER: 0.05,  // Just at or below shoulder
    SWING_MIN_HEIGHT_ABOVE_HIP: 0.1,  // Must go above hip (not a deadlift)

    // Configuration detection (only check when actually needed)
    TWO_HAND_PROXIMITY: 0.18,  // Both hands close together on one bell
    DOUBLE_BELL_MIN_SPACING: 0.25  // Both hands wide apart = two bells
  },

  // EXPORT
  MAKE_WEBHOOK_URL: "https://hook.us2.make.com/0l88dnosrk2t8a29yfk83fej8hp8j3jk",

  // DEBUGGING
  DEBUG_MODE: true
};

// ============================================
// STATE (Minimal additions to existing state)
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

  // Hand Tracking (EXISTING - already works)
  activeTrackingSide: "left",
  lockedSide: "unknown",
  armingSide: null,

  // Physics (EXISTING - already works)
  prevWrist: null,
  lockedCalibration: null,
  smoothedVelocity: 0,
  smoothedVy: 0,
  lastSpeed: 0,
  lastVy: 0,

  // Gesture Detection
  parkingConfirmed: false,
  prevHeadY: 0,

  // Rep Logic (EXISTING - already works)
  phase: "IDLE",
  currentRepPeak: 0,
  overheadHoldCount: 0,
  session: {
    currentSet: null,
    history: []
  },
  repHistory: [],
  baseline: 0,

  // ✅ NEW: Track peak position during rep (for classification)
  currentRepPeakWristY: 1.0,  // Normalized Y position at lockout
  currentRepPeakWristX: 0.5,  // Normalized X position at lockout

  // ✅ NEW: Movement history for display
  movementHistory: []
};

// ============================================
// INITIALIZATION (UNCHANGED)
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
// VIDEO INPUTS (UNCHANGED)
// ============================================

function handleUpload(e) {
  const file = e.target.files?.[0];
  if (!file) return;

  resetSession();
  setStatus("Loading Video...", "#fbbf24");
// Reset video to beginning
if (state.video && state.video.src) 
  state.video.pause();
  state.video.currentTime = 0;


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

    // ✅ Reset movement display
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
// MASTER LOOP (UNCHANGED - uses existing logic)
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

  // ✅ ALWAYS RUN PHYSICS (existing proven logic)
  if (state.isTestRunning) {
    runPhysics(pose, state.timeMs);

    if (state.testStage === "IDLE") {
      checkStartCondition(pose, state.timeMs);
    }

    if (state.testStage === "RUNNING") {
      runSnatchLogic(pose);  // ✅ This is your proven state machine
      checkEndCondition(pose, state.timeMs);
    }
  }

  drawOverlay();
}

// ============================================
// START/END CONDITIONS (UNCHANGED)
// ============================================

function checkStartCondition(pose, timeMs) {
  if (state.testStage !== "IDLE") return;

  const lWrist = pose[CONFIG.LEFT.WRIST];
  const rWrist = pose[CONFIG.RIGHT.WRIST];
  if (!lWrist || !rWrist) return;

  const lY = lWrist.y;
  const rY = rWrist.y;
  const activeSide = lY > rY ? "left" : "right";

  state.activeTrackingSide = activeSide;

  const inZone = isWristInFloorZone(pose, activeSide);
  const hikingDown = state.lastVy > 0.3 && state.lastSpeed > 0.5;

  if (CONFIG.DEBUG_MODE && inZone) {
    console.log(`[START] Side:${activeSide} | Zone:${inZone} | Hike:${hikingDown} | Vy:${state.lastVy.toFixed(2)} | Speed:${state.lastSpeed.toFixed(2)}`);
  }

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

  if (CONFIG.DEBUG_MODE && inZone) {
    console.log(`[END] Zone:${inZone} | StandUp:${standingUp} | Vy:${state.lastVy.toFixed(2)} | Speed:${state.lastSpeed.toFixed(2)}`);
  }

  if (inZone && standingUp) {
    console.log(`🛑 ENDING SET`);
    endCurrentSet();
  }
}

// ============================================
// SET MANAGEMENT (UNCHANGED)
// ============================================

function startNewSet(side) {
  state.lockedSide = side;
  state.testStage = "RUNNING";
  state.lockedAtMs = state.timeMs;

  state.repHistory = [];
  state.currentRepPeak = 0;
  state.overheadHoldCount = 0;
  state.baseline = 0;

  state.phase = "BOTTOM";

  state.session.currentSet = {
    id: state.session.history.length + 1,
    hand: side,
    reps: [],
    startTime: new Date(),
    lockedAtMs: state.timeMs
  };

  updateUIValues(0, 0, "--", "#fff");
  setStatus(`LOCKED: ${side.toUpperCase()}`, "#10b981");

  if (CONFIG.DEBUG_MODE) console.log(`🚀 Set Started [${side}]. Phase: BOTTOM.`);
}

function endCurrentSet() {
  if (state.session.currentSet) {
    state.session.currentSet.endTime = new Date();
    const peaks = state.session.currentSet.reps;

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
// PHYSICS ENGINE (UNCHANGED - Your proven code)
// ============================================

function runPhysics(pose, timeMs) {
  const side = state.testStage === "IDLE" ? state.activeTrackingSide : state.lockedSide;
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

  if (state.testStage === "RUNNING") {
    document.getElementById("val-velocity").textContent = state.lastSpeed.toFixed(2);
  }
}

// ============================================
// REP DETECTION (Enhanced to track peak position)
// ============================================

function runSnatchLogic(pose) {
  const v = state.smoothedVelocity;
  const vy = state.smoothedVy;
  const idx = state.lockedSide === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const hip = pose[idx.HIP];
  const shoulder = pose[idx.SHOULDER];

  if (!wrist || !hip || !shoulder) return;

  const isBelowHip = wrist.y > (hip.y - 0.05);
  const isAboveShoulder = wrist.y < shoulder.y;

  // --- STATE MACHINE (YOUR PROVEN LOGIC) ---

  // A. RESET
  if (state.phase === "IDLE" || state.phase === "LOCKOUT") {
    if (isBelowHip) {
      state.phase = "BOTTOM";
      state.overheadHoldCount = 0;
      // ✅ Reset peak position tracking
      state.currentRepPeakWristY = 1.0;
      state.currentRepPeakWristX = wrist.x;
      if (CONFIG.DEBUG_MODE) console.log("Phase: BOTTOM (Ready to Pull)");
    }
  }

  // B. PULL
  else if (state.phase === "BOTTOM") {
    if (!isBelowHip && vy < -0.4) {
      state.phase = "CONCENTRIC";
      state.currentRepPeak = 0;
      if (CONFIG.DEBUG_MODE) console.log("Phase: CONCENTRIC (Pulling)");
    }
  }

  // C. POWER
  else if (state.phase === "CONCENTRIC") {
    // 1. Capture peak velocity (EXISTING)
    if (v > state.currentRepPeak) state.currentRepPeak = v;

    // ✅ 2. Track HIGHEST position reached (for movement classification)
    if (wrist.y < state.currentRepPeakWristY) {
      state.currentRepPeakWristY = wrist.y;
      state.currentRepPeakWristX = wrist.x;
    }

    // 3. Lockout Check (EXISTING PROVEN LOGIC)
    const isStable = Math.abs(vy) < CONFIG.LOCKOUT_VY_CUTOFF && v < CONFIG.LOCKOUT_SPEED_CUTOFF;

    if (isAboveShoulder && isStable) {
      state.overheadHoldCount++;

      if (state.overheadHoldCount >= 2) {
        recordRep(pose);  // ✅ Pass pose for classification
        if (CONFIG.DEBUG_MODE) console.log(`📊 Rep ${state.repHistory.length} Recorded!`);
      }
    } else {
      state.overheadHoldCount = 0;

      if (isBelowHip) {
        state.phase = "BOTTOM";
        if (CONFIG.DEBUG_MODE) console.log("Phase: BOTTOM (Aborted/Dropped)");
      }
    }
  }
}

// ============================================
// ✅ MOVEMENT CLASSIFICATION (Uses Proven Criteria)
// ============================================

function classifyMovement(pose) {
  if (!pose) return 'UNKNOWN';

  const idx = state.lockedSide === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const shoulder = pose[idx.SHOULDER];
  const hip = pose[idx.HIP];

  if (!shoulder || !hip) return 'UNKNOWN';

  const peakY = state.currentRepPeakWristY;
  const peakX = state.currentRepPeakWristX;

  // Calculate relative height to shoulder
  const heightAboveShoulder = shoulder.y - peakY;
  const heightAboveHip = hip.y - peakY;

  // Calculate torso centerline (for clean detection)
  const lShoulder = pose[CONFIG.LEFT.SHOULDER];
  const rShoulder = pose[CONFIG.RIGHT.SHOULDER];
  const torsoCenter = lShoulder && rShoulder ? (lShoulder.x + rShoulder.x) / 2 : shoulder.x;
  const horizontalDistFromCenter = Math.abs(peakX - torsoCenter);

  if (CONFIG.DEBUG_MODE) {
    console.log(`[CLASSIFY] PeakY:${peakY.toFixed(3)} | ShoulderY:${shoulder.y.toFixed(3)} | HeightAboveShoulder:${heightAboveShoulder.toFixed(3)} | HorizDist:${horizontalDistFromCenter.toFixed(3)}`);
  }

  // ✅ SNATCH: Locked out OVERHEAD (clearly above shoulder)
  // This uses the SAME criteria as your existing snatch detection
  if (heightAboveShoulder >= CONFIG.MOVEMENT.SNATCH_MIN_HEIGHT_ABOVE_SHOULDER) {
    return `SNATCH_SINGLE_${state.lockedSide.toUpperCase()}`;
  }

  // ✅ CLEAN: Locked at RACK position (at shoulder height, close to torso)
  if (heightAboveShoulder >= CONFIG.MOVEMENT.CLEAN_RACK_HEIGHT_MIN &&
      heightAboveShoulder <= CONFIG.MOVEMENT.CLEAN_RACK_HEIGHT_MAX &&
      horizontalDistFromCenter < CONFIG.MOVEMENT.CLEAN_HORIZONTAL_PROXIMITY) {
    return `CLEAN_SINGLE_${state.lockedSide.toUpperCase()}`;
  }

  // ✅ SWING: Locked at or below shoulder height (didn't go overhead)
  if (heightAboveShoulder < CONFIG.MOVEMENT.SWING_MAX_HEIGHT_ABOVE_SHOULDER &&
      heightAboveHip >= CONFIG.MOVEMENT.SWING_MIN_HEIGHT_ABOVE_HIP) {
    return `SWING_SINGLE_${state.lockedSide.toUpperCase()}`;
  }

  // If it doesn't match any criteria clearly
  if (CONFIG.DEBUG_MODE) {
    console.log(`[CLASSIFY] UNKNOWN - Criteria not met`);
  }
  return `UNKNOWN_SINGLE_${state.lockedSide.toUpperCase()}`;
}

function recordRep(pose) {
  state.phase = "LOCKOUT";
  state.overheadHoldCount = 0;

  // ✅ Classify movement using PROVEN lockout criteria
  const movementType = classifyMovement(pose);

  if (state.session.currentSet) {
    state.session.currentSet.reps.push({
      velocity: state.currentRepPeak,
      movement: movementType
    });
  }

  state.repHistory.push(state.currentRepPeak);

  // ✅ Update movement history
  state.movementHistory.push(movementType);
  if (state.movementHistory.length > 3) {
    state.movementHistory.shift();
  }

  // Calculate baseline
  if (state.repHistory.length === CONFIG.BASELINE_REPS) {
    state.baseline = state.repHistory.reduce((a,b) => a+b, 0) / CONFIG.BASELINE_REPS;
  }

  // Calculate drop-off
  let dropPct = "--";
  let dropColor = "#fff";

  if (state.baseline > 0 && state.repHistory.length > CONFIG.BASELINE_REPS) {
    const drop = ((state.baseline - state.currentRepPeak) / state.baseline) * 100;
    dropPct = drop.toFixed(1) + "%";

    if (drop < CONFIG.DROP_WARN) {
      dropColor = "#10b981";
    } else if (drop < CONFIG.DROP_FAIL) {
      dropColor = "#fbbf24";
    } else {
      dropColor = "#ef4444";
    }
  }

  updateUIValues(state.repHistory.length, state.currentRepPeak, dropPct, dropColor);

  // ✅ Update movement display
  updateMovementDisplay(movementType);

  if (CONFIG.DEBUG_MODE) {
    console.log(`📊 REP #${state.repHistory.length}: ${state.currentRepPeak.toFixed(2)} m/s | Drop: ${dropPct} | Movement: ${movementType}`);
  }
}

// ============================================
// HELPERS (UNCHANGED)
// ============================================

function isWristInFloorZone(pose, side) {
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const knee = pose[idx.KNEE];
  if (!wrist || !knee) return false;
  return wrist.y > knee.y;
}

function resetSession() {
  // Clear session data
  state.session = { currentSet: null, history: [] };
  state.repHistory = [];
  state.baseline = 0;
  
  // Reset test stage
  state.testStage = "IDLE";
  state.lockedSide = "unknown";
  state.activeTrackingSide = "left";
  state.armingSide = null;
  
  // Reset physics
  state.smoothedVelocity = 0;
  state.smoothedVy = 0;
  state.lastSpeed = 0;
  state.lastVy = 0;
  state.lockedCalibration = null;
  state.prevWrist = null;
  
  // Reset rep detection
  state.phase = "IDLE";
  state.currentRepPeak = 0;
  state.overheadHoldCount = 0;
  state.parkingConfirmed = false;
  
  // Reset movement tracking
  state.currentRepPeakWristY = 1.0;
  state.currentRepPeakWristX = 0.5;
  state.movementHistory = [];
  
  // Reset ending confirmation
  state.endingConfirmCount = 0;
  
  // Reset video to beginning
  if (state.video && state.video.src) {
    state.video.pause();
    state.video.currentTime = 0;
  }
  
  // Reset UI
  updateUIValues(0, 0, "--", "#fff");
  resetMovementDisplay();
  setStatus("Session Cleared — Ready", "#3b82f6");
  
  if (CONFIG.DEBUG_MODE) {
    console.log("🔄 Session Reset - All state cleared");
  }
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

// ✅ Movement Display Functions (Simplified)

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
  }

  // Configuration is always "Single Kettlebell" for now (hand already shown)
  configEl.textContent = 'Single Kettlebell';

  // Active hand comes from locked side
  handsEl.textContent = state.lockedSide === 'left' ? 'Left' : state.lockedSide === 'right' ? 'Right' : '—';

  statusIndicator.className = 'status-indicator locked';

  updateMovementHistory(movementType);
}

function formatMovementName(movementType) {
  const typeMap = {
    'SWING_SINGLE_LEFT': 'One Arm Swing (L)',
    'SWING_SINGLE_RIGHT': 'One Arm Swing (R)',
    'CLEAN_SINGLE_LEFT': 'One Arm Clean (L)',
    'CLEAN_SINGLE_RIGHT': 'One Arm Clean (R)',
    'SNATCH_SINGLE_LEFT': 'One Arm Snatch (L)',
    'SNATCH_SINGLE_RIGHT': 'One Arm Snatch (R)',
    'UNKNOWN_SINGLE_LEFT': 'Unknown Movement (L)',
    'UNKNOWN_SINGLE_RIGHT': 'Unknown Movement (R)'
  };
  return typeMap[movementType] || movementType.replace(/_/g, ' ');
}

function updateMovementHistory(movementType) {
  const historyContainer = document.getElementById('history-badges');
  if (!historyContainer) return;

  const badge = document.createElement('span');
  badge.className = 'history-badge';
  badge.textContent = formatMovementName(movementType).replace(/\s*\(.\)/, '');

  historyContainer.insertBefore(badge, historyContainer.firstChild);

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

  if (CONFIG.DEBUG_MODE) {
    state.ctx.fillStyle = "#fbbf24";
    state.ctx.font = "12px monospace";
    state.ctx.fillText(`Side: ${state.testStage === "IDLE" ? state.activeTrackingSide : state.lockedSide}`, 10, 20);
    state.ctx.fillText(`Speed: ${state.lastSpeed.toFixed(2)} m/s`, 10, 35);
    state.ctx.fillText(`Vy: ${state.lastVy.toFixed(2)} m/s`, 10, 50);
    state.ctx.fillText(`Stage: ${state.testStage}`, 10, 65);
    state.ctx.fillText(`Phase: ${state.phase}`, 10, 80);

    // ✅ Show peak position being tracked
    if (state.phase === "CONCENTRIC" || state.phase === "LOCKOUT") {
      state.ctx.fillText(`PeakY: ${state.currentRepPeakWristY.toFixed(3)}`, 10, 95);
    }
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
        movements: movements
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
