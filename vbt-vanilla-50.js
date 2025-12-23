/**
 * VBT Vanilla 50 - Simplified Movement Tracker
 * Basic velocity tracking without complex movement classification
 */

import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

// ============================================
// CONFIG
// ============================================

const CONFIG = {
  // Landmark indices
  WRIST: 16,        // Right wrist
  SHOULDER: 12,     // Right shoulder
  HIP: 24,          // Right hip
  KNEE: 26,         // Right knee
  
  // Calibration
  TORSO_METERS: 0.45,
  
  // Physics
  SMOOTHING_ALPHA: 0.2,
  MIN_DT: 0.016,
  MAX_DT: 0.1,
  
  // Rep detection
  REP_START_THRESHOLD: 0.5,     // m/s - moving down
  REP_END_THRESHOLD: 0.3,       // m/s - slowing down at top
  MIN_REP_HEIGHT: 0.2,          // meters above starting position
  
  // MediaPipe
  MIN_DET_CONF: 0.5,
  MIN_TRACK_CONF: 0.5
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
  isTracking: false,
  
  // Physics
  prevWrist: null,
  calibration: null,
  velocity: 0,
  smoothedVy: 0,
  
  // Rep tracking
  isInRep: false,
  repStartY: 0,
  currentRepPeak: 0,
  totalReps: 0,
  repHistory: [],
  sessionPeak: 0,
  
  timeMs: 0
};

// ============================================
// INITIALIZATION
// ============================================

async function initializeApp() {
  state.video = document.getElementById("video");
  state.canvas = document.getElementById("canvas");
  state.ctx = state.canvas.getContext("2d");
  
  // Event listeners
  document.getElementById("btn-camera").onclick = startCamera;
  document.getElementById("file-input").onchange = handleUpload;
  document.getElementById("btn-start").onclick = toggleTracking;
  document.getElementById("btn-reset").onclick = resetSession;
  document.getElementById("btn-export").onclick = exportData;
  
  // Load MediaPipe model
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
  setStatus("Ready - Upload Video or Start Camera");
  
  state.video.addEventListener("loadeddata", onVideoReady);
  requestAnimationFrame(mainLoop);
}

// ============================================
// VIDEO INPUTS
// ============================================

function handleUpload(e) {
  const file = e.target.files?.[0];
  if (!file) return;
  
  state.video.srcObject = null;
  state.video.src = URL.createObjectURL(file);
  state.video.load();
  setStatus("Video Loaded");
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false
  });
  
  state.video.srcObject = stream;
  state.video.src = "";
  state.video.play();
  setStatus("Camera Active");
}

function onVideoReady() {
  state.canvas.width = state.video.videoWidth;
  state.canvas.height = state.video.videoHeight;
  document.getElementById("btn-start").disabled = false;
  
  if (state.video.src) {
    state.video.pause();
    state.video.currentTime = 0;
  }
  
  setStatus("Ready to Start");
}

function toggleTracking() {
  if (!state.isTracking) {
    state.isTracking = true;
    document.getElementById("btn-start").textContent = "Pause";
    setStatus("Tracking Active");
    
    if (state.video.paused) {
      state.video.play().catch(() => {});
    }
  } else {
    state.isTracking = false;
    document.getElementById("btn-start").textContent = "Resume";
    setStatus("Paused");
    state.video.pause();
  }
}

// ============================================
// MAIN LOOP
// ============================================

async function mainLoop(timestamp) {
  requestAnimationFrame(mainLoop);
  
  if (!state.isModelLoaded || !state.video) return;
  
  state.timeMs = timestamp;
  
  // Draw video frame
  state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
  state.ctx.drawImage(state.video, 0, 0, state.canvas.width, state.canvas.height);
  
  // Detect pose
  let pose = null;
  if (state.landmarker && state.video.readyState >= 2) {
    try {
      const results = state.landmarker.detectForVideo(state.video, timestamp);
      if (results?.landmarks?.[0]) {
        pose = results.landmarks[0];
      }
    } catch (e) {
      console.warn("Detection error:", e);
    }
  }
  
  if (!pose) {
    drawOverlay("No pose detected");
    return;
  }
  
  if (state.isTracking) {
    processPhysics(pose, timestamp);
    detectReps(pose);
    updateUI();
  }
  
  drawOverlay(pose);
}

// ============================================
// PHYSICS
// ============================================

function processPhysics(pose, timeMs) {
  const wrist = pose[CONFIG.WRIST];
  const shoulder = pose[CONFIG.SHOULDER];
  const hip = pose[CONFIG.HIP];
  
  if (!wrist || !shoulder || !hip) return;
  
  // Calibrate
  if (!state.calibration) {
    const torsoPx = Math.abs(shoulder.y - hip.y) * state.canvas.height;
    state.calibration = Math.max(50, torsoPx / CONFIG.TORSO_METERS);
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
  
  // Calculate velocity
  const dyPx = (wrist.y - state.prevWrist.y) * state.canvas.height;
  let vy = (dyPx / state.calibration) / dt;
  
  // Normalize to 30 FPS
  const TARGET_FPS = 30;
  const frameTimeMs = 1000 / TARGET_FPS;
  const actualFrameTimeMs = timeMs - state.prevWrist.t;
  const timeRatio = frameTimeMs / actualFrameTimeMs;
  vy *= timeRatio;
  
  // Smooth
  state.smoothedVy = CONFIG.SMOOTHING_ALPHA * vy + (1 - CONFIG.SMOOTHING_ALPHA) * state.smoothedVy;
  state.velocity = Math.abs(state.smoothedVy);
  
  state.prevWrist = { x: wrist.x, y: wrist.y, t: timeMs };
}

// ============================================
// REP DETECTION
// ============================================

function detectReps(pose) {
  const wrist = pose[CONFIG.WRIST];
  if (!wrist) return;
  
  const vy = state.smoothedVy;
  const speed = state.velocity;
  
  // Start of rep - moving downward
  if (!state.isInRep && vy > CONFIG.REP_START_THRESHOLD) {
    state.isInRep = true;
    state.repStartY = wrist.y;
    state.currentRepPeak = 0;
  }
  
  // During rep - track peak velocity
  if (state.isInRep) {
    if (speed > state.currentRepPeak) {
      state.currentRepPeak = speed;
    }
    
    // End of rep - slowing down after moving up
    const heightMoved = state.repStartY - wrist.y;
    const heightMeters = (heightMoved * state.canvas.height) / state.calibration;
    
    if (vy < -CONFIG.REP_END_THRESHOLD && heightMeters > CONFIG.MIN_REP_HEIGHT && speed < CONFIG.REP_END_THRESHOLD) {
      recordRep();
    }
  }
}

function recordRep() {
  state.totalReps++;
  
  const repData = {
    number: state.totalReps,
    velocity: state.currentRepPeak,
    timestamp: new Date().toISOString()
  };
  
  state.repHistory.push(repData);
  
  if (state.currentRepPeak > state.sessionPeak) {
    state.sessionPeak = state.currentRepPeak;
  }
  
  // Add to UI
  addRepToList(repData);
  
  // Reset for next rep
  state.isInRep = false;
  state.currentRepPeak = 0;
  
  console.log(`Rep #${state.totalReps}: ${repData.velocity.toFixed(2)} m/s`);
}

// ============================================
// UI UPDATES
// ============================================

function updateUI() {
  document.getElementById("total-reps").textContent = state.totalReps;
  document.getElementById("velocity").textContent = state.velocity.toFixed(2);
  document.getElementById("peak-velocity").textContent = state.sessionPeak.toFixed(2);
}

function addRepToList(repData) {
  const repList = document.getElementById("rep-list");
  const repItem = document.createElement("div");
  repItem.className = "rep-item";
  repItem.innerHTML = `
    <span>Rep #${repData.number}</span>
    <span>${repData.velocity.toFixed(2)} m/s</span>
  `;
  repList.insertBefore(repItem, repList.firstChild);
}

function setStatus(text) {
  const pill = document.getElementById("status-pill");
  if (pill) pill.textContent = text;
}

// ============================================
// SESSION MANAGEMENT
// ============================================

function resetSession() {
  state.totalReps = 0;
  state.repHistory = [];
  state.sessionPeak = 0;
  state.isInRep = false;
  state.currentRepPeak = 0;
  state.velocity = 0;
  state.smoothedVy = 0;
  state.prevWrist = null;
  state.calibration = null;
  
  document.getElementById("rep-list").innerHTML = "";
  updateUI();
  
  if (state.video?.src) {
    state.video.pause();
    state.video.currentTime = 0;
  }
  
  setStatus("Session Reset");
  console.log("Session reset");
}

function exportData() {
  if (state.repHistory.length === 0) {
    alert("No reps to export");
    return;
  }
  
  const data = {
    sessionDate: new Date().toISOString(),
    totalReps: state.totalReps,
    peakVelocity: state.sessionPeak,
    reps: state.repHistory
  };
  
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement("a");
  a.href = url;
  a.download = `vbt-vanilla-50-${new Date().toISOString().split('T')[0]}.json`;
  a.click();
  
  URL.revokeObjectURL(url);
  
  console.log("Data exported:", data);
  setStatus("Data Exported");
}

// ============================================
// DRAWING
// ============================================

function drawOverlay(pose) {
  if (!pose || typeof pose === "string") {
    // Draw status text if no pose
    state.ctx.fillStyle = "#4ecca3";
    state.ctx.font = "16px sans-serif";
    state.ctx.fillText(typeof pose === "string" ? pose : "No pose", 10, 30);
    return;
  }
  
  // Draw wrist tracking
  const wrist = pose[CONFIG.WRIST];
  if (wrist) {
    const x = wrist.x * state.canvas.width;
    const y = wrist.y * state.canvas.height;
    
    state.ctx.beginPath();
    state.ctx.fillStyle = state.isInRep ? "#10b981" : "#4ecca3";
    state.ctx.arc(x, y, 10, 0, 2 * Math.PI);
    state.ctx.fill();
  }
  
  // Draw velocity info
  state.ctx.fillStyle = "#4ecca3";
  state.ctx.font = "14px monospace";
  state.ctx.fillText(`Velocity: ${state.velocity.toFixed(2)} m/s`, 10, 30);
  state.ctx.fillText(`Reps: ${state.totalReps}`, 10, 50);
  
  if (state.isInRep) {
    state.ctx.fillStyle = "#10b981";
    state.ctx.fillText("IN REP", 10, 70);
  }
}

// ============================================
// START APP
// ============================================

initializeApp();
