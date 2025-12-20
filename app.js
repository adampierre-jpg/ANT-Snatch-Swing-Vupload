import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

class VBTStateMachine {
  constructor(canvasHeight = 720) {
    this.canvasHeight = canvasHeight;
    this.THRESHOLDS = {
      HINGE: 0.12, // More lenient hinge detection
      RACK_LOCK_FRAMES: 15, // Faster rack detection
      PULL_VELOCITY_TRIGGER: 0.4,
      LOCKOUT_VY_CUTOFF: 0.5,
      CLEAN_HOLD_FRAMES: 20,  // 0.6s hold is enough
      CLEAN_HOLD_VY: 0.3,
      VELOCITY_ALPHA: 0.2,
      POSITION_ALPHA: 0.4,
      TORSO_METERS: 0.45,
      RESET_DURATION_FRAMES: 30
    };
    this.calibrationData = { isCalibrated: false, framesCaptured: 0, neutralWristOffset: 0, maxTorsoLength: 0 };
    this.reset();
  }

  reset() {
    this.state = {
      currentPose: "NONE",
      rackFrameCount: 0,
      phase: "IDLE",
      movementStartPose: "NONE", // KEY: Captured at start of pull
      currentRepPeak: 0,
      shoulderHoldFrames: 0, 
      smoothedVy: 0,
      lastWristPos: null,
      calibration: null,
      smoothedLandmarks: {
        LEFT: { WRIST: null, SHOULDER: null, HIP: null, KNEE: null, NOSE: null },
        RIGHT: { WRIST: null, SHOULDER: null, HIP: null, KNEE: null, NOSE: null }
      }
    };
  }

  smoothLandmarks(rawPose) {
    const alpha = this.THRESHOLDS.POSITION_ALPHA;
    const smoothed = { LEFT: {}, RIGHT: {} };
    for (const side of ['LEFT', 'RIGHT']) {
      for (const landmark of ['WRIST', 'SHOULDER', 'HIP', 'KNEE', 'NOSE', 'ELBOW']) {
        if (!rawPose[side] || !rawPose[side][landmark]) continue;
        const raw = rawPose[side][landmark];
        const prev = this.state.smoothedLandmarks[side]?.[landmark];
        if (!prev) smoothed[side][landmark] = { x: raw.x, y: raw.y, z: raw.z || 0 };
        else smoothed[side][landmark] = {
            x: alpha * raw.x + (1 - alpha) * prev.x,
            y: alpha * raw.y + (1 - alpha) * prev.y,
            z: alpha * (raw.z || 0) + (1 - alpha) * (prev.z || 0)
        };
      }
    }
    this.state.smoothedLandmarks = smoothed;
    return smoothed;
  }

  calculateVelocity(wrist, timestamp) {
    if (!this.state.lastWristPos || !this.state.calibration) {
      this.state.lastWristPos = { x: wrist.x, y: wrist.y, t: timestamp };
      return { vy: 0 };
    }
    const dt = (timestamp - this.state.lastWristPos.t) / 1000;
    if (dt < 0.01 || dt > 0.1) return { vy: 0 };
    const dyPx = (wrist.y - this.state.lastWristPos.y) * this.canvasHeight;
    let vy = (dyPx / this.state.calibration) / dt;
    this.state.lastWristPos = { x: wrist.x, y: wrist.y, t: timestamp };
    return { vy };
  }

  update(pose, timestamp, ctx, canvas) {
    if (!pose.LEFT || !pose.RIGHT) return null;
    const smoothedPose = this.smoothLandmarks(pose);
    
    // --- Calibration ---
    if (!this.calibrationData.isCalibrated) {
      this.calibrationData.framesCaptured++;
      if (this.calibrationData.framesCaptured >= 30) this.calibrationData.isCalibrated = true;
      return null; 
    }

    // --- Side Selection ---
    if (this.state.lockedSide === "unknown") {
      if (Math.abs(smoothedPose.LEFT.WRIST.y - smoothedPose.RIGHT.WRIST.y) > 0.1) 
        this.state.lockedSide = smoothedPose.LEFT.WRIST.y > smoothedPose.RIGHT.WRIST.y ? "LEFT" : "RIGHT";
      else return null;
    }

    const side = this.state.lockedSide;
    const wrist = smoothedPose[side].WRIST;
    const hip = smoothedPose[side].HIP;
    const shoulder = smoothedPose[side].SHOULDER;
    const nose = smoothedPose[side].NOSE;

    if (!this.state.calibration) 
      this.state.calibration = (Math.abs(shoulder.y - hip.y) * this.canvasHeight) / this.THRESHOLDS.TORSO_METERS;

    const velocity = this.calculateVelocity(wrist, timestamp);
    this.state.smoothedVy = (this.THRESHOLDS.VELOCITY_ALPHA * velocity.vy) + ((1 - this.THRESHOLDS.VELOCITY_ALPHA) * this.state.smoothedVy);
    
    // --- Current Pose Detection ---
    const hinged = wrist.y > hip.y - 0.05; 
    const atRack = Math.abs(wrist.y - shoulder.y) < 0.12 && Math.abs(wrist.x - shoulder.x) < 0.2;

    if (atRack) {
      if (++this.state.rackFrameCount >= this.THRESHOLDS.RACK_LOCK_FRAMES) this.state.currentPose = "RACK";
    } else {
      this.state.rackFrameCount = 0;
      if (hinged) this.state.currentPose = "HINGE";
      else if (this.state.phase === "IDLE") this.state.currentPose = "NONE";
    }

    // --- State Machine ---
    let result = null;
    if (this.state.phase === "IDLE") {
      if (this.state.smoothedVy < -this.THRESHOLDS.PULL_VELOCITY_TRIGGER) {
        this.state.phase = "PULLING";
        this.state.movementStartPose = this.state.currentPose; // LOCK THE STARTING STATE
        this.state.currentRepPeak = 0;
        this.state.shoulderHoldFrames = 0;
      }
    } else if (this.state.phase === "PULLING") {
      this.state.currentRepPeak = Math.max(this.state.currentRepPeak, Math.abs(this.state.smoothedVy));
      const isAtShoulder = Math.abs(wrist.y - shoulder.y) < 0.12;
      const nearlyStopped = Math.abs(this.state.smoothedVy) < this.THRESHOLDS.LOCKOUT_VY_CUTOFF;
      
      if (nearlyStopped && isAtShoulder) {
        if (++this.state.shoulderHoldFrames >= this.THRESHOLDS.CLEAN_HOLD_FRAMES) {
            result = this.classify(wrist, shoulder, nose, this.state.movementStartPose, true);
            this.state.phase = "LOCKED";
        }
      } else if (nearlyStopped) {
        result = this.classify(wrist, shoulder, nose, this.state.movementStartPose, false);
        this.state.phase = "LOCKED";
      }
      if (result) result.velocity = this.state.currentRepPeak;
    } else if (this.state.phase === "LOCKED") {
      // Allow next rep once movement starts again or returns to hinge/rack
      if (Math.abs(this.state.smoothedVy) > this.THRESHOLDS.PULL_VELOCITY_TRIGGER) this.state.phase = "IDLE";
    }
    return result;
  }

  classify(w, s, nose, startPose, held) {
    const isOverhead = w.y < (nose.y - 0.05); 
    
    // If we started at the Rack, it's a Press.
    if (startPose === "RACK") {
        if (isOverhead) return { type: "PRESS" };
        return null; 
    }
    
    // If we started from a Hinge/Swing, it's a Ballistic.
    if (isOverhead) return { type: "SNATCH" };
    if (held) return { type: "CLEAN" };
    return { type: "SWING" };
  }
}

const app = {
  video: null, canvas: null, ctx: null, landmarker: null, stateMachine: null,
  isModelLoaded: false, isTestRunning: false, totalReps: 0, lastMove: "READY",
  history: { CLEAN: [], PRESS: [], SNATCH: [], SWING: [] }
};

async function initializeApp() {
  app.video = document.getElementById("video");
  app.canvas = document.getElementById("canvas");
  app.ctx = app.canvas.getContext("2d");
  document.getElementById("btn-camera").onclick = startCamera;
  document.getElementById("file-input").onchange = handleUpload;
  document.getElementById("btn-start-test").onclick = toggleTest;
  document.getElementById("btn-reset").onclick = resetSession;
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm");
  app.landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task", delegate: "GPU" },
    runningMode: "VIDEO"
  });
  app.isModelLoaded = true;
  requestAnimationFrame(masterLoop);
}

async function startCamera() {
  const s = await navigator.mediaDevices.getUserMedia({ video: true });
  app.video.srcObject = s;
  app.video.onloadedmetadata = () => {
    app.canvas.width = app.video.videoWidth; app.canvas.height = app.video.videoHeight;
    app.stateMachine = new VBTStateMachine(app.canvas.height);
    document.getElementById("btn-start-test").disabled = false;
  };
}

function handleUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    app.video.src = URL.createObjectURL(file);
    app.video.onloadedmetadata = () => {
      app.canvas.width = app.video.videoWidth; app.canvas.height = app.video.videoHeight;
      app.stateMachine = new VBTStateMachine(app.canvas.height);
      document.getElementById("btn-start-test").disabled = false;
    };
}

function toggleTest() {
  app.isTestRunning = !app.isTestRunning;
  document.getElementById("btn-start-test").innerText = app.isTestRunning ? "PAUSE" : "START";
  if (app.isTestRunning) app.video.play(); else app.video.pause();
}

async function masterLoop(ts) {
  requestAnimationFrame(masterLoop);
  if (!app.isModelLoaded || !app.video.readyState) return;
  app.ctx.drawImage(app.video, 0, 0, app.canvas.width, app.canvas.height);
  const results = app.landmarker.detectForVideo(app.video, ts);
  if (results?.landmarks?.length > 0) {
    const raw = results.landmarks[0];
    const pose = {
      LEFT: { WRIST: raw[15], SHOULDER: raw[11], HIP: raw[23], KNEE: raw[25], NOSE: raw[0], ELBOW: raw[13] },
      RIGHT: { WRIST: raw[16], SHOULDER: raw[12], HIP: raw[24], KNEE: raw[26], NOSE: raw[0], ELBOW: raw[14] }
    };
    if (app.isTestRunning && app.stateMachine) {
      const move = app.stateMachine.update(pose, ts, app.ctx, app.canvas);
      if (move) record(move);
      drawUI(app.stateMachine.state);
      drawDebugSkeleton(pose);
    }
  }
}

function drawDebugSkeleton(pose) {
  const ctx = app.ctx; const canvas = app.canvas;
  for (const side of ['LEFT', 'RIGHT']) {
    const color = side === 'LEFT' ? '#00ff00' : '#ff0000';
    const joints = ['WRIST', 'ELBOW', 'SHOULDER', 'HIP', 'KNEE'];
    ctx.strokeStyle = color; ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(pose[side].WRIST.x * canvas.width, pose[side].WRIST.y * canvas.height);
    joints.forEach(j => ctx.lineTo(pose[side][j].x * canvas.width, pose[side][j].y * canvas.height));
    ctx.stroke();
    joints.forEach(j => {
        ctx.fillStyle = color; ctx.beginPath();
        ctx.arc(pose[side][j].x * canvas.width, pose[side][j].y * canvas.height, 6, 0, Math.PI*2);
        ctx.fill();
    });
  }
}

function record(m) {
  app.totalReps++; app.lastMove = m.type; app.history[m.type].push(m.velocity);
  let plural = m.type.toLowerCase() + "es";
  if (m.type === "SWING") plural = "swings";
  if (m.type === "CLEAN") plural = "cleans";
  const countEl = document.getElementById(`val-${plural}`);
  const velEl = document.getElementById(`val-${m.type.toLowerCase()}-velocity`);
  if (countEl) countEl.innerText = app.history[m.type].length;
  if (velEl) velEl.innerText = m.velocity.toFixed(2);
  document.getElementById("val-total-reps").innerText = app.totalReps;
  document.getElementById("detected-movement").innerText = m.type;
}

function resetSession() {
  app.totalReps = 0; app.lastMove = "READY";
  app.history = { CLEAN: [], PRESS: [], SNATCH: [], SWING: [] };
  if (app.stateMachine) app.stateMachine.reset();
  ['val-cleans', 'val-presses', 'val-snatches', 'val-swings', 'val-total-reps'].forEach(id => {
      const el = document.getElementById(id); if (el) el.textContent = '0';
  });
  document.getElementById("detected-movement").innerText = "READY";
}

function drawUI(s) {
  document.getElementById("val-velocity").innerText = Math.abs(s.smoothedVy).toFixed(2);
}

initializeApp();
