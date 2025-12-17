/**
 * Version 3.1 — "Snap-Park" Logic
 * 
 * - START: 0.5s dwell (Deliberate) - "Lowest Hand" wins.
 * - END: 0.15s dwell (Fast) - Guarded by "Not Pulling Up" check.
 * - FIX: Prevents missing "Touch-and-Go" parks while ignoring hikes.
 */

import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const CONFIG = {
  // Your existing LEFT/RIGHT landmark indexes...
  LEFT: {
    WRIST: 15,
    SHOULDER: 11,
    HIP: 23,
    // ... etc
  },
  RIGHT: {
    WRIST: 16,
    SHOULDER: 12,
    HIP: 24,
    // ... etc
  },
  
  // ✅ ADD THESE:
  HEAD_LANDMARK: 0,              // Nose for head tracking
  TORSO_METERS: 0.45,            // Your existing torso calibration
  SMOOTHING_ALPHA: 0.15,         // Heavy smoothing for velocity
  MAX_REALISTIC_VELOCITY: 8.0,   // Cap outliers
  ZERO_BAND: 0.1,                // Dead zone
  MIN_DT: 0.016,                 // ~60fps min
  MAX_DT: 0.1,                   // Skip lag spikes
  RESET_GRACE_MS_AFTER_LOCK: 2000,

  

  // Update these in CONFIG:
LOCKOUT_VY_CUTOFF: 0.35,  // Was 0.40 (Stricter vertical stop)
LOCKOUT_SPEED_CUTOFF: 1.0, // Was 1.40 (Must be very still overhead)


  // Drop Feedback
  BASELINE_REPS: 3,
  DROP_WARN: 15,
  DROP_FAIL: 20,

  // Floor/Zone Logic
  MIN_SHANK_LEN_NORM: 0.06,
  
  // --- HANDSHAKE LOGIC ---
  ARM_MS_REQUIRED: 500,        // Start: 0.5s (Deliberate)
  RESET_MS_REQUIRED: 150,      // End: 0.15s (Snap-Park)
  STILLNESS_THRESHOLD_START: 1.5, // Start: Permissive
  STILLNESS_THRESHOLD_END: 0.8,   // End: Strict (Must be stopped)
  RESET_GRACE_MS_AFTER_LOCK: 1500, // Buffer before you can end

   MAKE_WEBHOOK_URL: "https://hook.us2.make.com/bxyeuukaw4v71k32vx26jwiqbumgi19c"
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
  armingSide: null,
  lockedAtMs: 0,
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
  document.getElementById("btn-camera").onclick = startCamera;
  document.getElementById("file-input").onchange = handleUpload;
  document.getElementById("btn-start-test").onclick = toggleTest;
  document.getElementById("btn-reset").onclick = resetSession;
const saveBtn = document.getElementById("btn-save");
  if (saveBtn) saveBtn.onclick = exportToMake;
  
  // ... rest of init ...
  const visionGen = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  
  
  
}

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

  document.getElementById("btn-camera").onclick = startCamera;
  document.getElementById("file-input").onchange = handleUpload;
  document.getElementById("btn-start-test").onclick = toggleTest;
  document.getElementById("btn-reset").onclick = resetSession;
  
  state.video.addEventListener("loadeddata", onVideoReady);
  
  requestAnimationFrame(masterLoop);
}

// --- INPUTS ---

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

// --- MASTER LOOP ---

function masterLoop(timestamp) {
  requestAnimationFrame(masterLoop);
  
  if (!state.isModelLoaded || !state.video) return;
  
  state.timeMs = timestamp;
  
  // Draw video frame
  state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
  state.ctx.drawImage(state.video, 0, 0, state.canvas.width, state.canvas.height);
  
  // Detect pose
  let pose = null;
  if (state.detector) {
    const results = await state.detector.estimatePoses(state.canvas);
    if (results && results.length > 0) {
      pose = results[0].keypoints;
      drawSkeleton(pose);  // Your existing draw function
    }
  }
  
  if (!pose) return;
  
  // ✅ RUN ALL STATE MACHINES
  runPhysics(pose, state.timeMs);
  
  if (state.testStage === "IDLE") {
    checkStartCondition(pose, state.timeMs);
  }
  
  if (state.testStage === "RUNNING") {
    checkEndCondition(pose, state.timeMs);
    // Your existing rep detection logic here
  }
}


  drawOverlay();
  requestAnimationFrame(masterLoop);
}

// --- LOGIC: START / END ---

function checkStartCondition(pose, timeMs) {
  if (state.testStage !== "IDLE") return;
  
  const lY = pose[CONFIG.LEFT.WRIST]?.y || 0;
  const rY = pose[CONFIG.RIGHT.WRIST]?.y || 0;
  const activeSide = lY > rY ? "left" : "right";  // Dominant low hand
  
  const sideIdx = activeSide === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[sideIdx.WRIST];
  const head = pose[CONFIG.HEAD_LANDMARK];  // Nose Y
  if (!wrist || !head) return;
  
  const inZone = isWristInFloorZone(pose, activeSide);
  const headLowering = head.y > state.prevHeadY + 0.02;  // Dip = ready position
  const hikingDown = state.lastVy > 0.4 && state.lastSpeed > 0.6;  // Pull backswing
  
  state.prevHeadY = head.y;
  
  // Park confirm: Floor + dip
  if (inZone && headLowering) {
    state.parkingConfirmed = true;
    state.armingSide = activeSide;  // Auto-arm
  }
  
  // Start trigger: Confirmed + hike
  if (state.parkingConfirmed && inZone && hikingDown) {
    startNewSet(state.armingSide);
    state.parkingConfirmed = false;
    state.prevHeadY = 0;
  }
}



function checkEndCondition(pose, timeMs) {
  if (state.testStage !== "RUNNING") return;
  if (!state.session.currentSet || (timeMs - state.session.currentSet.lockedAtMs < CONFIG.RESET_GRACE_MS_AFTER_LOCK)) return;
  
  const sideIdx = state.lockedSide === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[sideIdx.WRIST];
  const head = pose[CONFIG.HEAD_LANDMARK];  // Nose Y
  if (!wrist || !head) return;
  
  const inZone = isWristInFloorZone(pose, state.lockedSide);  // Your existing func
  const headLowering = head.y > state.prevHeadY + 0.02;      // Dip threshold
  const standingUp = state.lastVy < -0.4 && state.lastSpeed > 0.6;  // Up + moving
  
  state.prevHeadY = head.y;  // Update every frame
  
  // Park confirm: Floor zone + head dip (handles slow/quick)
  if (inZone && headLowering) {
    state.parkingConfirmed = true;
  }
  
  // End trigger: Confirmed park + standing
  if (state.parkingConfirmed && inZone && standingUp) {
    endCurrentSet();
    state.parkingConfirmed = false;
    state.prevHeadY = 0;  // Reset
  }
}



function startNewSet(side) {
  state.lockedSide = side;
  state.testStage = "RUNNING";
  state.lockedAtMs = state.lastLoopMs;
  state.dwellTimerMs = 0;
  
  // Reset Set Data
  state.repHistory = [];
  state.currentRepPeak = 0;
  state.overheadHoldCount = 0;
  state.phase = "IDLE";
  
  // Create Session
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
    
    // Calculate Average Velocity for this set (for easy reading in Make/Notion)
    const peaks = state.session.currentSet.reps;
    const avg = peaks.length > 0 ? peaks.reduce((a,b)=>a+b,0)/peaks.length : 0;
    state.session.currentSet.avgVelocity = avg.toFixed(2);
    
    // Push the finalized set to session history
    state.session.history.push(state.session.currentSet);
  }
  
  // Reset System State
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
  if (wrist.y < 0 || wrist.y > 1.0) return false;
  return wrist.y > knee.y; 
}

function calculatePassiveVelocity(pose, timeMs) {
  // Track Lowest Hand for IDLE velocity
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

  // Stable Calibration (min 50px guard)
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

  // Frame-Rate Normalization (fixes video replay jitter)
  const TARGET_FPS = 30;
  const frameTimeMs = 1000 / TARGET_FPS;
  const actualFrameTimeMs = timeMs - state.prevWrist.t;
  const timeRatio = frameTimeMs / actualFrameTimeMs;
  vx *= timeRatio;
  vy *= timeRatio;
  speed = Math.hypot(vx, vy);

  // Zero Band
  if (speed < CONFIG.ZERO_BAND) speed = 0;

  // Heavy Smoothing
  state.smoothedVelocity = CONFIG.SMOOTHING_ALPHA * speed + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVelocity;
  state.smoothedVy = CONFIG.SMOOTHING_ALPHA * vy + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVy;
  
  // Velocity Ceiling
  state.lastSpeed = Math.min(state.smoothedVelocity, CONFIG.MAX_REALISTIC_VELOCITY);
  state.lastVy = Math.min(Math.max(state.smoothedVy, -CONFIG.MAX_REALISTIC_VELOCITY), CONFIG.MAX_REALISTIC_VELOCITY);
  
  state.prevWrist = { x: wrist.x, y: wrist.y, t: timeMs };
  
  if (state.testStage === "RUNNING") {
    document.getElementById("val-velocity").textContent = state.lastSpeed.toFixed(2);
  }
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

  // ✅ FRAME-RATE NORMALIZATION (Fixes video playback jitter)
  const TARGET_FPS = 30;
  const frameTimeMs = 1000 / TARGET_FPS;
  const actualFrameTimeMs = timeMs - state.prevWrist.t;
  const timeRatio = frameTimeMs / actualFrameTimeMs;
  vx *= timeRatio;
  vy *= timeRatio;
  speed = Math.hypot(vx, vy);

  // ZERO BAND
  if (speed < CONFIG.ZERO_BAND) speed = 0;

  // ✅ HEAVY SMOOTHING
  state.smoothedVelocity = CONFIG.SMOOTHING_ALPHA * speed + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVelocity;
  state.smoothedVy = CONFIG.SMOOTHING_ALPHA * vy + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVy;
  
  // ✅ VELOCITY CEILING
  state.lastSpeed = Math.min(state.smoothedVelocity, CONFIG.MAX_REALISTIC_VELOCITY);
  state.lastVy = Math.min(Math.max(state.smoothedVy, -CONFIG.MAX_REALISTIC_VELOCITY), CONFIG.MAX_REALISTIC_VELOCITY);
  
  state.prevWrist = { x: wrist.x, y: wrist.y, t: timeMs };
  
  if (state.testStage === "RUNNING") {
    document.getElementById("val-velocity").textContent = state.lastSpeed.toFixed(2);
  }
}


// --- REPLACES CURRENT runSnatchLogic() ---

function runSnatchLogic() {
  const v = state.smoothedVelocity;
  const vy = state.smoothedVy;
  const pose = state.lastPose;
  
  // Handlers
  const idx = state.lockedSide === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const hip = pose[idx.HIP];
  const shoulder = pose[idx.SHOULDER];
  const nose = pose[0]; // Nose is always index 0

  // 1. ZONES
  // Note: Y is normalized (0=top, 1=bottom). Smaller Y is higher.
  const isBelowHip = wrist.y > hip.y;
  const isAboveShoulder = wrist.y < shoulder.y;
  const isAboveNose = wrist.y < nose.y; // Strict Height Check

  // 2. STATE MACHINE
  if (state.phase === "IDLE" || state.phase === "LOCKOUT") {
    // Reset when bell drops below hip (Hike/Backswing)
    if (isBelowHip) {
      state.phase = "BOTTOM";
      state.overheadHoldCount = 0;
    }
  } 
  else if (state.phase === "BOTTOM") {
    // Trigger Concentric phase when bell passes hip upward
    if (!isBelowHip) {
      state.phase = "CONCENTRIC";
      state.currentRepPeak = 0; // Reset peak for this rep
    }
  } 
  else if (state.phase === "CONCENTRIC") {
    // A. Track Velocity (ONLY while below shoulder to capture hip power)
    if (!isAboveShoulder) { 
        if (v > state.currentRepPeak) state.currentRepPeak = v;
    }

    // B. Check for Lockout
    // Criteria: Above Nose + Low Vertical Speed + Low Total Speed
    const isStable = Math.abs(vy) < CONFIG.LOCKOUT_VY_CUTOFF && v < CONFIG.LOCKOUT_SPEED_CUTOFF;
    
    if (isAboveNose && isStable) {
      // Must hold for ~200ms (6 frames @ 30fps)
      state.overheadHoldCount++;
      if (state.overheadHoldCount >= 2) { // Was 6
        recordRep();
      }
      }
    
    }
  }



function recordRep() {
  state.phase = "LOCKOUT";
  state.overheadHoldCount = 0;
  
  // Clean Data: Only push the single peak value for this rep
  if (state.session.currentSet) {
      state.session.currentSet.reps.push(state.currentRepPeak);
  }
  
  // Local history for display
  state.repHistory.push(state.currentRepPeak);
  
  // Update UI
  updateUIValues(state.repHistory.length, state.currentRepPeak);
}

// --- VISUALS ---

function updateUIValues(reps, peak, drop, dropColor) {
  document.getElementById("val-reps").textContent = reps;
  document.getElementById("val-peak").textContent = peak.toFixed(2);
  const d = document.getElementById("val-drop");
  d.textContent = drop;
  if (dropColor) d.style.color = dropColor;
}

function resetSession() {
  // Clear Session Data
  state.session = { currentSet: null, history: [] };
  state.repHistory = [];
  
  // Reset State Machine
  state.testStage = "IDLE";
  state.lockedSide = "unknown";
  state.armingSide = null;
  state.dwellTimerMs = 0;
  
  // Reset Physics
  state.smoothedVelocity = 0;
  state.smoothedVy = 0;
  state.lastSpeed = 0;
  state.lastVy = 0;
  state.lockedCalibration = null;
  state.prevWrist = null;
  
  // Reset Gesture Tracking
  state.parkingConfirmed = false;
  state.prevHeadY = 0;
  state.gestureState = { phase: null, side: null };
  
  // Reset UI
  updateUIValues(0, 0);
  setStatus("Session Cleared — Ready", "#3b82f6");
  
  console.log("Session Reset Complete");
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
  
  // --- DEBUG OVERLAY (REMOVE IF NEEDED) ---
  state.ctx.fillStyle = "#fbbf24";
  state.ctx.font = "14px monospace";
  state.ctx.fillText(`Timer: ${(state.dwellTimerMs/1000).toFixed(2)}s`, 10, 20);
  state.ctx.fillText(`Speed: ${state.lastSpeed.toFixed(2)}`, 10, 40);
  state.ctx.fillText(`Vy: ${state.lastVy.toFixed(2)}`, 10, 60);
  // ----------------------------------------

  const side = state.lockedSide;

  if (state.testStage === "RUNNING") {
    const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
    const wrist = state.lastPose[idx.WRIST];
    drawDot(wrist, true, "#10b981");
    
    // Draw parking line (green if criteria met)
    const inZone = isWristInFloorZone(state.lastPose, side);
    const isStill = state.lastSpeed < CONFIG.STILLNESS_THRESHOLD_END;
    const isNotPulling = state.lastVy > -0.5;
    const readyToPark = inZone && isStill && isNotPulling;
    
    drawParkingLine(state.lastPose, side, readyToPark);
  } else {
    const lY = state.lastPose[CONFIG.LEFT.WRIST].y;
    const rY = state.lastPose[CONFIG.RIGHT.WRIST].y;
    const lowest = lY > rY ? "left" : "right";
    
    const isStill = state.lastSpeed < CONFIG.STILLNESS_THRESHOLD_START;
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
async function exportToMake() {
  const history = state.session.history;
  
  if (!history.length) {
    alert("No completed sets to export.");
    return;
  }

  // 1. Calculate Session-Level Totals
  const totalReps = history.reduce((sum, set) => sum + set.reps.length, 0);
  // Avoid division by zero
  const totalAvg = history.length > 0 ? 
      (history.reduce((sum, set) => sum + parseFloat(set.avgVelocity), 0) / history.length) : 0;

  // 2. Construct the Make.com Payload
  const payload = {
    athlete_id: "dad_ready_user", // You can make this dynamic later
    session_date: new Date().toISOString(),
    total_reps: totalReps,
    session_avg_velocity: totalAvg.toFixed(2),
    sets: history.map((set, index) => ({
      set_order: index + 1,
      hand: set.hand,
      rep_count: set.reps.length,
      peak_velocity_avg: parseFloat(set.avgVelocity),
      raw_peaks: set.reps // Array of peak velocities [1.2, 1.4, ...]
    }))
  };

  // 3. Log to Console (Debugging)
  console.log("--------------------------------");
  console.log("EXPORTING TO MAKE (PAYLOAD):");
  console.log(JSON.stringify(payload, null, 2));
  console.log("--------------------------------");

  setStatus("Exporting to Make...", "#8b5cf6");

  // 4. Send to Webhook (Uncomment when you have the URL)
  /*
  try {
    const response = await fetch(CONFIG.MAKE_WEBHOOK_URL, {
       method: "POST",
       headers: { "Content-Type": "application/json" },
       body: JSON.stringify(payload)
    });
    
    if(response.ok) {
        setStatus("Success! Session Saved.", "#10b981");
        // Optional: resetSession(); // Clear data after save
    } else {
        setStatus("Error: Make.com rejected payload.", "#ef4444");
    }
  } catch(e) {
      console.error("Network Error:", e);
      setStatus("Network Error (Check Console)", "#ef4444");
  }
  */


}


initializeApp();


