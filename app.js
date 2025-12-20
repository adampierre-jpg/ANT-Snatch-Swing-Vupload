import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

class VBTStateMachine {
  constructor(canvasHeight = 720) {
    this.canvasHeight = canvasHeight;
    this.THRESHOLDS = {
      HINGE_DEPTH: 0.05, // CHANGED: Back to sensitive detection
      RACK_LOCK_FRAMES: 30,
      PULL_VELOCITY_TRIGGER: 0.4,
      LOCKOUT_VY_CUTOFF: 0.6,
      CLEAN_HOLD_FRAMES: 30,
      SHOULDER_ZONE: 0.12,
      SWING_HEIGHT_BUFFER: 0.08,
      VELOCITY_ALPHA: 0.15,
      POSITION_ALPHA: 0.3,
      TORSO_METERS: 0.45,
      RESET_DURATION_FRAMES: 30,
      MAX_REALISTIC_VELOCITY: 8.0,
      MIN_DT: 0.016,
      MAX_DT: 0.1
    };

    this.calibrationData = { isCalibrated: false, framesCaptured: 0, neutralWristOffset: 0, maxTorsoLength: 0 };
    this.reset();
  }

  reset() {
    this.state = {
      lockedSide: "unknown",
      phase: "IDLE",
      currentRepPeak: 0,
      hipCrossedUpward: false,
      hasVisitedHinge: false,
      shoulderHoldFrames: 0, 
      smoothedVy: 0,
      lastWristPos: null,
      calibration: null,
      resetProgress: 0,
      smoothedLandmarks: {
        LEFT: { WRIST: null, SHOULDER: null, HIP: null, KNEE: null, NOSE: null, ELBOW: null },
        RIGHT: { WRIST: null, SHOULDER: null, HIP: null, KNEE: null, NOSE: null, ELBOW: null }
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
        const prev = this.state.smoothedLandmarks[side][landmark];
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
      return { vx: 0, vy: 0, speed: 0 };
    }
    const dt = (timestamp - this.state.lastWristPos.t) / 1000;
    if (dt < this.THRESHOLDS.MIN_DT || dt > this.THRESHOLDS.MAX_DT) {
      this.state.lastWristPos = { x: wrist.x, y: wrist.y, t: timestamp };
      return { vx: 0, vy: 0, speed: 0 };
    }
    const dxPx = (wrist.x - this.state.lastWristPos.x) * this.canvasHeight;
    const dyPx = (wrist.y - this.state.lastWristPos.y) * this.canvasHeight;
    let vx = (dxPx / this.state.calibration) / dt;
    let vy = (dyPx / this.state.calibration) / dt;
    const TARGET_FPS = 30;
    const timeRatio = (1000 / TARGET_FPS) / (timestamp - this.state.lastWristPos.t);
    vx *= timeRatio; vy *= timeRatio;
    this.state.lastWristPos = { x: wrist.x, y: wrist.y, t: timestamp };
    return { vx, vy, speed: Math.hypot(vx, vy) };
  }

  update(pose, timestamp, ctx, canvas) {
    if (!pose.LEFT || !pose.RIGHT) return null;
    const smoothedPose = this.smoothLandmarks(pose);
    const currentTorso = Math.abs(smoothedPose.LEFT.SHOULDER.y - smoothedPose.LEFT.HIP.y);
    const leftWristOffset = smoothedPose.LEFT.WRIST.y - smoothedPose.LEFT.HIP.y;
    const rightWristOffset = smoothedPose.RIGHT.WRIST.y - smoothedPose.RIGHT.HIP.y;

    if (!this.calibrationData.isCalibrated) {
      this.calibrationData.framesCaptured++;
      this.calibrationData.neutralWristOffset += (leftWristOffset + rightWristOffset) / 2;
      this.calibrationData.maxTorsoLength = Math.max(this.calibrationData.maxTorsoLength, currentTorso);
      if (this.calibrationData.framesCaptured >= 30) {
        this.calibrationData.neutralWristOffset /= 30;
        this.calibrationData.isCalibrated = true;
      }
      return null; 
    }

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

    if (!this.state.calibration && shoulder && hip) 
      this.state.calibration = (Math.abs(shoulder.y - hip.y) * this.canvasHeight) / this.THRESHOLDS.TORSO_METERS;

    const velocity = this.calculateVelocity(wrist, timestamp);
    this.state.smoothedVy = (this.THRESHOLDS.VELOCITY_ALPHA * velocity.vy) + ((1 - this.THRESHOLDS.VELOCITY_ALPHA) * this.state.smoothedVy);
    
    // Track hinge visit
    if (wrist.y > hip.y + this.THRESHOLDS.HINGE_DEPTH) {
        this.state.hasVisitedHinge = true;
    }

    const hinged = Math.abs(shoulder.y - hip.y) < 0.08;

    let result = null;
    if (this.state.phase === "IDLE") {
      if (this.state.smoothedVy < -this.THRESHOLDS.PULL_VELOCITY_TRIGGER) {
        this.state.phase = "PULLING";
        this.state.currentRepPeak = 0;
        this.state.hipCrossedUpward = false;
        this.state.shoulderHoldFrames = 0;
      }
    } else if (this.state.phase === "PULLING") {
      this.state.currentRepPeak = Math.max(this.state.currentRepPeak, Math.abs(this.state.smoothedVy));
      if (wrist.y < hip.y) this.state.hipCrossedUpward = true;

      const isAtShoulder = wrist.y <= (shoulder.y + this.THRESHOLDS.SHOULDER_ZONE) && wrist.y >= (shoulder.y - this.THRESHOLDS.SHOULDER_ZONE);
      const nearlyStopped = Math.abs(this.state.smoothedVy) < this.THRESHOLDS.LOCKOUT_VY_CUTOFF;
      const isOverhead = wrist.y < (nose.y - 0.05);
      
      // CLEAN: Hold at shoulder for 1 second
      if (nearlyStopped && isAtShoulder && !hinged && !isOverhead) {
        this.state.shoulderHoldFrames++;
        if (this.state.shoulderHoldFrames >= this.THRESHOLDS.CLEAN_HOLD_FRAMES) {
            if (this.state.hasVisitedHinge && this.state.hipCrossedUpward) {
                result = { type: "CLEAN", velocity: this.state.currentRepPeak };
            }
            this.state.phase = "LOCKED";
        }
      } 
      // LOCKOUT: Overhead or at shoulder without hold
      else if (nearlyStopped) {
        if (isOverhead) {
            // Overhead = Press or Snatch
            if (this.state.hasVisitedHinge && this.state.hipCrossedUpward) {
                result = { type: "SNATCH", velocity: this.state.currentRepPeak };
            } else {
                result = { type: "PRESS", velocity: this.state.currentRepPeak };
            }
        } else if (this.state.hasVisitedHinge && this.state.hipCrossedUpward) {
            // At shoulder but didn't hold = Swing
            result = { type: "SWING", velocity: this.state.currentRepPeak };
        }
        this.state.phase = "LOCKED";
      }
    } else if (this.state.phase === "LOCKED") {
      if (Math.abs(this.state.smoothedVy) > this.THRESHOLDS.PULL_VELOCITY_TRIGGER) {
          this.state.phase = "IDLE";
          this.state.hasVisitedHinge = false;
      }
    }
    
    this.drawSkeleton(ctx, canvas, smoothedPose, side);
    return result;
  }

  drawSkeleton(ctx, canvas, pose, activeSide) {
    for (const side of ['LEFT', 'RIGHT']) {
      const color = side === activeSide ? '#00ff00' : '#ff0000';
      const w = pose[side].WRIST;
      const e = pose[side].ELBOW;
      const s = pose[side].SHOULDER;
      const h = pose[side].HIP;
      const k = pose[side].KNEE;
      
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(w.x * canvas.width, w.y * canvas.height);
      ctx.lineTo(e.x * canvas.width, e.y * canvas.height);
      ctx.lineTo(s.x * canvas.width, s.y * canvas.height);
      ctx.lineTo(h.x * canvas.width, h.y * canvas.height);
      ctx.lineTo(k.x * canvas.width, k.y * canvas.height);
      ctx.stroke();
      
      [w, e, s, h, k].forEach(pt => {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 5, 0, Math.PI * 2);
          ctx.fill();
      });
    }
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

async function startCamera() {
  const s = await navigator.mediaDevices.getUserMedia({ video: true });
  app.video.srcObject = s;
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
  app.ctx.clearRect(0, 0, app.canvas.width, app.canvas.height);
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
    }
  }
}

function record(m) {
  app.totalReps++; app.lastMove = m.type; app.history[m.type].push(m.velocity);
  let plural = m.type.toLowerCase() + "s";
  if (m.type === "PRESS") plural = "presses";
  if (m.type === "SNATCH") plural = "snatches";
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
  ['val-clean-velocity', 'val-press-velocity', 'val-snatch-velocity', 'val-swing-velocity', 'val-velocity'].forEach(id => {
      const el = document.getElementById(id); if (el) el.textContent = '0.00';
  });
  document.getElementById("detected-movement").innerText = "READY";
}

function drawUI(s) {
  const velEl = document.getElementById("val-velocity");
  if (velEl) velEl.innerText = Math.abs(s.smoothedVy).toFixed(2);
}

initializeApp();
