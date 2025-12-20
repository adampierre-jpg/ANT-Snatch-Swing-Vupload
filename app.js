import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

class VBTStateMachine {
  constructor(canvasHeight = 720) {
    this.canvasHeight = canvasHeight;
    this.THRESHOLDS = {
      HINGE: 0.08,
      KNEE_CROSS: 0.05,
      RACK_HEIGHT_MIN: -0.1,
      RACK_HEIGHT_MAX: 0.15,
      RACK_HORIZONTAL_PROXIMITY: 0.18,
      RACK_LOCK_FRAMES: 15,
      OVERHEAD_MIN_HEIGHT: 0.05,
      PULL_VELOCITY_TRIGGER: 0.4,
      LOCKOUT_VY_CUTOFF: 0.6,
      CLEAN_HOLD_FRAMES: 15,  
      CLEAN_HOLD_VY: 0.3,     
      SWING_HEIGHT_BUFFER: 0.08,
      VELOCITY_ALPHA: 0.15,
      POSITION_ALPHA: 0.3,
      TORSO_METERS: 0.45,
      RESET_DURATION_FRAMES: 30,
      MAX_REALISTIC_VELOCITY: 8.0,
      ZERO_BAND: 0.1,
      MIN_DT: 0.016,
      MAX_DT: 0.1
    };

    this.calibrationData = {
      isCalibrated: false,
      framesCaptured: 0,
      neutralWristOffset: 0, 
      maxTorsoLength: 0      
    };

    this.reset();
  }

  reset() {
    this.state = {
      currentPose: "NONE",
      rackFrameCount: 0,
      lockedSide: "unknown",
      phase: "IDLE",
      movementStartPose: null,
      currentRepPeak: 0,
      hipCrossedUpward: false,
      wristStartedBelowHip: false,
      shoulderHoldFrames: 0, 
      smoothedVy: 0,
      lastTimestamp: 0,
      lastWristY: null,
      lastWristPos: null,
      calibration: null,
      resetProgress: 0,
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
      for (const landmark of ['WRIST', 'SHOULDER', 'HIP', 'KNEE', 'NOSE']) {
        if (!rawPose[side] || !rawPose[side][landmark]) continue;
        
        const raw = rawPose[side][landmark];
        const prev = this.state.smoothedLandmarks[side][landmark];

        if (!prev) {
          smoothed[side][landmark] = { x: raw.x, y: raw.y, z: raw.z || 0 };
        } else {
          smoothed[side][landmark] = {
            x: alpha * raw.x + (1 - alpha) * prev.x,
            y: alpha * raw.y + (1 - alpha) * prev.y,
            z: alpha * (raw.z || 0) + (1 - alpha) * (prev.z || 0)
          };
        }
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

    const TARGET_FPS = 15;
    const frameTimeMs = 500 / TARGET_FPS;
    const actualFrameTimeMs = timestamp - this.state.lastWristPos.t;
    const timeRatio = frameTimeMs / actualFrameTimeMs;
    
    vx *= timeRatio;
    vy *= timeRatio;
    
    let speed = Math.hypot(vx, vy);
    
    if (speed < this.THRESHOLDS.ZERO_BAND) {
      speed = 0;
      vx = 0;
      vy = 0;
    }

    speed = Math.min(speed, this.THRESHOLDS.MAX_REALISTIC_VELOCITY);
    vy = Math.min(Math.max(vy, -this.THRESHOLDS.MAX_REALISTIC_VELOCITY), this.THRESHOLDS.MAX_REALISTIC_VELOCITY);
    
    this.state.lastWristPos = { x: wrist.x, y: wrist.y, t: timestamp };
    
    return { vx, vy, speed };
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
        console.log("✅ Calibration Complete");
      }
      return null; 
    }

    const leftAtHome = Math.abs(leftWristOffset - this.calibrationData.neutralWristOffset) < 0.10;
    const rightAtHome = Math.abs(rightWristOffset - this.calibrationData.neutralWristOffset) < 0.10;
    const isTall = currentTorso > (this.calibrationData.maxTorsoLength * 0.85);

    if (leftAtHome && rightAtHome && isTall) {
      this.state.resetProgress++;
      this.drawResetUI(ctx, canvas, smoothedPose);

      if (this.state.resetProgress > this.THRESHOLDS.RESET_DURATION_FRAMES) {
        this.reset();
        return null;
      }
    } else {
      this.state.resetProgress = 0;
    }

    if (this.state.lockedSide === "unknown") {
      if (Math.abs(smoothedPose.LEFT.WRIST.y - smoothedPose.RIGHT.WRIST.y) > 0.1) {
        this.state.lockedSide = smoothedPose.LEFT.WRIST.y > smoothedPose.RIGHT.WRIST.y ? "LEFT" : "RIGHT";
      } else {
        return null;
      }
    }

    const side = this.state.lockedSide;
    const wrist = smoothedPose[side].WRIST;
    const hip = smoothedPose[side].HIP;
    const shoulder = smoothedPose[side].SHOULDER;
    const nose = smoothedPose[side].NOSE;

    if (!this.state.calibration && shoulder && hip) {
      this.state.calibration = (Math.abs(shoulder.y - hip.y) * this.canvasHeight) / this.THRESHOLDS.TORSO_METERS;
    }

    const velocity = this.calculateVelocity(wrist, timestamp);
    
    this.state.smoothedVy = (this.THRESHOLDS.VELOCITY_ALPHA * velocity.vy) + 
                            ((1 - this.THRESHOLDS.VELOCITY_ALPHA) * this.state.smoothedVy);
    
    this.state.lastTimestamp = timestamp;
    this.state.lastWristY = wrist.y;

    const hinged = Math.abs(shoulder.y - hip.y) < this.THRESHOLDS.HINGE;
    const atRack = Math.abs(wrist.y - shoulder.y) < 0.15 && 
                   Math.abs(wrist.x - (smoothedPose.LEFT.SHOULDER.x + smoothedPose.RIGHT.SHOULDER.x)/2) < 0.2;

    // RACK DETECTION - Only runs when NOT actively pulling
    if (this.state.phase === "IDLE" || this.state.phase === "LOCKED") {
      if (atRack && !hinged) {
        if (++this.state.rackFrameCount >= this.THRESHOLDS.RACK_LOCK_FRAMES) 
          this.state.currentPose = "RACK";
      } else {
        this.state.rackFrameCount = 0;
        if (wrist.y > (smoothedPose[side].KNEE.y + this.THRESHOLDS.KNEE_CROSS)) 
          this.state.currentPose = "HINGE";
        else if (this.state.phase === "IDLE") 
          this.state.currentPose = "NONE";
      }
    }

    let result = null;
    if (this.state.phase === "IDLE") {
      if (this.state.smoothedVy < -this.THRESHOLDS.PULL_VELOCITY_TRIGGER) {
        this.state.phase = "PULLING";
        this.state.movementStartPose = this.state.currentPose;
        this.state.currentRepPeak = 0;
        this.state.hipCrossedUpward = false;
        this.state.wristStartedBelowHip = wrist.y > hip.y;
        this.state.shoulderHoldFrames = 0;
      }
    } else if (this.state.phase === "PULLING") {
      this.state.currentRepPeak = Math.max(this.state.currentRepPeak, Math.abs(this.state.smoothedVy));
      
      if (wrist.y < hip.y) this.state.hipCrossedUpward = true;

      const isAtShoulder = wrist.y <= (shoulder.y + 0.12) && wrist.y >= (shoulder.y - 0.08);
      const nearlyStopped = Math.abs(this.state.smoothedVy) < this.THRESHOLDS.LOCKOUT_VY_CUTOFF;
      
      // CLEAN DETECTION: Hold at shoulder for 30 frames
      if (nearlyStopped && isAtShoulder && !hinged) {
        this.state.shoulderHoldFrames++;
        if (this.state.shoulderHoldFrames >= this.THRESHOLDS.CLEAN_HOLD_FRAMES) {
            result = this.classify(this.state.movementStartPose, wrist, shoulder, hip, nose, hinged, this.state.hipCrossedUpward, true);
            this.state.phase = "LOCKED";
            
            // AUTO-SET RACK: If clean finished, bell is already at rack
            if (result && result.type === "CLEAN" && atRack) {
              this.state.currentPose = "RACK";
              this.state.rackFrameCount = this.THRESHOLDS.RACK_LOCK_FRAMES;
            }
        }
      } else if (nearlyStopped) {
        // Quick lockout for other movements
        result = this.classify(this.state.movementStartPose, wrist, shoulder, hip, nose, hinged, this.state.hipCrossedUpward, false);
        this.state.phase = "LOCKED";
      } else if (this.state.smoothedVy > this.THRESHOLDS.PULL_VELOCITY_TRIGGER && this.state.shoulderHoldFrames > 0) {
        // Bell started falling while at shoulder without completing hold
        result = this.classify(this.state.movementStartPose, wrist, shoulder, hip, nose, hinged, this.state.hipCrossedUpward, false);
        this.state.phase = "LOCKED";
      }

      if (result) result.velocity = this.state.currentRepPeak;
    } else if (this.state.phase === "LOCKED") {
      if (wrist.y > hip.y || atRack) {
        this.state.phase = "IDLE";
      }
    }

    return result;
  }

  classify(start, w, s, h, nose, hinged, crossed, heldAtShoulder) {
    const isOverhead = w.y < (nose.y - 0.05); 
    
    // CLEAN from hinge/standing
    if ((start === "HINGE" || start === "NONE") && crossed) {
      if (isOverhead) return { type: "SNATCH" };
      
      // CLEAN: Held at shoulder for 30 frames, within 8% of shoulder height
      if (heldAtShoulder && w.y <= (s.y + 0.08)) return { type: "CLEAN" };
      
      // SWING: Reached shoulder area but didn't hold, stayed below hip
      if (w.y <= (s.y + this.THRESHOLDS.SWING_HEIGHT_BUFFER) && w.y < h.y) return { type: "SWING" };
    }
    
    // RECLEAN: Started from rack, held at shoulder
    if (start === "RACK" && heldAtShoulder) {
      return { type: "CLEAN" };
    }
    
    // PRESS: Started from rack, went overhead
    if (start === "RACK" && isOverhead) {
      return { type: "PRESS" };
    }
    
    return null;
  }

  drawResetUI(ctx, canvas, pose) {
    const centerX = (pose.LEFT.SHOULDER.x + pose.RIGHT.SHOULDER.x) / 2 * canvas.width;
    const centerY = (pose.LEFT.SHOULDER.y + pose.LEFT.HIP.y) / 2 * canvas.height;
    const pct = this.state.resetProgress / this.THRESHOLDS.RESET_DURATION_FRAMES;
    ctx.beginPath(); ctx.arc(centerX, centerY, 40, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255,255,255,0.2)"; ctx.lineWidth = 8; ctx.stroke();
    ctx.beginPath(); ctx.arc(centerX, centerY, 40, -Math.PI/2, (-Math.PI/2) + (Math.PI * 2 * pct));
    ctx.strokeStyle = "#3b82f6"; ctx.stroke();
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
  app.ctx.drawImage(app.video, 0, 0, app.canvas.width, app.canvas.height);
  const results = app.landmarker.detectForVideo(app.video, ts);
  if (results?.landmarks?.length > 0) {
    const raw = results.landmarks[0];
    const pose = {
      LEFT: { 
        WRIST: raw[15], 
        SHOULDER: raw[11], 
        HIP: raw[23], 
        KNEE: raw[25], 
        ANKLE: raw[27],  // Added ankle
        NOSE: raw[0], 
        ELBOW: raw[13] 
      },
      RIGHT: { 
        WRIST: raw[16], 
        SHOULDER: raw[12], 
        HIP: raw[24], 
        KNEE: raw[26], 
        ANKLE: raw[28],  // Added ankle
        NOSE: raw[0], 
        ELBOW: raw[14] 
      }
    };
    if (app.isTestRunning && app.stateMachine) {
      const move = app.stateMachine.update(pose, ts, app.ctx, app.canvas);
      if (move) record(move);
      drawUI(app.stateMachine.state, pose);
      drawDebugSkeleton(pose);
    }
  }
}

function drawDebugSkeleton(pose) {
  const ctx = app.ctx; 
  const canvas = app.canvas;
  
  // Draw skeleton lines for both sides
  for (const side of ['LEFT', 'RIGHT']) {
    const color = side === 'LEFT' ? '#00ff00' : '#ff0000';
    const wrist = pose[side].WRIST; 
    const elbow = pose[side].ELBOW; 
    const shoulder = pose[side].SHOULDER; 
    const hip = pose[side].HIP; 
    const knee = pose[side].KNEE;
    const ankle = pose[side].ANKLE;  // Now drawing to ankle
    
    ctx.strokeStyle = color; 
    ctx.lineWidth = 3; 
    ctx.beginPath();
    ctx.moveTo(wrist.x * canvas.width, wrist.y * canvas.height);
    ctx.lineTo(elbow.x * canvas.width, elbow.y * canvas.height);
    ctx.lineTo(shoulder.x * canvas.width, shoulder.y * canvas.height);
    ctx.lineTo(hip.x * canvas.width, hip.y * canvas.height);
    ctx.lineTo(knee.x * canvas.width, knee.y * canvas.height);
    ctx.lineTo(ankle.x * canvas.width, ankle.y * canvas.height);  // Extended to ankle
    ctx.stroke();
  }
  
  // Draw 🙂 emoji at nose position
  const nose = pose.LEFT.NOSE;
  const leftShoulder = pose.LEFT.SHOULDER;
  const rightShoulder = pose.RIGHT.SHOULDER;
  
  // Calculate head size based on shoulder width (natural proportions)
  const shoulderWidth = Math.abs(leftShoulder.x - rightShoulder.x) * canvas.width;
  const headSize = shoulderWidth * 1.25;  // Head is roughly 35% of shoulder width
  
  ctx.font = `${headSize}px Arial`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('🙂', nose.x * canvas.width, nose.y * canvas.height);
}

function record(m) {
  app.totalReps++; 
  app.lastMove = m.type; 
  app.history[m.type].push(m.velocity);
  
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
      const el = document.getElementById(id);
      if (el) el.textContent = '0';
  });
  ['val-clean-velocity', 'val-press-velocity', 'val-snatch-velocity', 'val-swing-velocity', 'val-velocity'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = '0.00';
  });
  document.getElementById("detected-movement").innerText = "READY";
}

function drawUI(s, p) {
  document.getElementById("val-velocity").innerText = Math.abs(s.smoothedVy).toFixed(2);
}

initializeApp();
