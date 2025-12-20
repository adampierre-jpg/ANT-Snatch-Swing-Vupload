/* VBT Calibration System v4.0 - Real Body Measurements */
import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

class VBTStateMachine {
  constructor(canvasHeight = 720) {
    this.canvasHeight = canvasHeight;

    this.THRESHOLDS = {
      HINGE: 0.08,
      KNEECROSS: 0.05,
      RACKHEIGHTMIN: -0.1,
      RACKHEIGHTMAX: 0.15,
      RACKHORIZONTALPROXIMITY: 0.18,
      RACKLOCKFRAMES: 30,
      OVERHEADMINHEIGHT: 0.05,
      PULLVELOCITYTRIGGER: 0.4,
      LOCKOUTVYCUTOFF: 0.6,
      VELOCITYALPHA: 0.15,
      RESETDURATIONFRAMES: 30, // 1 second at 30fps
      NOSETOHEADCM: 11 // Average offset from nose to top of head
    };

    // Enhanced calibration data
    this.calibrationData = {
      isCalibrated: false,
      framesCaptured: 0,
      neutralWristOffset: 0,

      // Pixel measurements (accumulated over 30 frames)
      ankleToNosePixels: 0,
      shoulderToHipPixels: 0,
      hipToKneePixels: 0,
      kneeToAnklePixels: 0,
      shoulderToElbowPixels: 0,
      elbowToWristPixels: 0,

      // Real measurements in cm (calculated after calibration)
      pixelToCmRatio: 0,
      torsoLengthCm: 0,
      thighLengthCm: 0,
      shinLengthCm: 0,
      upperArmLengthCm: 0,
      forearmLengthCm: 0,
      fullHeightCm: 0,

      userHeightInches: 0
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
      smoothedVy: 0,
      lastTimestamp: 0,
      lastWristY: null,
      calibration: null,
      resetProgress: 0
    };
  }

  setUserHeight(inches) {
    this.calibrationData.userHeightInches = inches;
    this.calibrationData.fullHeightCm = inches * 2.54;
    console.log(`User height set: ${inches}" = ${this.calibrationData.fullHeightCm.toFixed(1)} cm`);
  }

  update(pose, timestamp, ctx, canvas) {
    if (!pose.LEFT || !pose.RIGHT) return null;

    const currentTorso = Math.abs(pose.LEFT.SHOULDER.y - pose.LEFT.HIP.y);
    const leftWristOffset = pose.LEFT.WRIST.y - pose.LEFT.HIP.y;
    const rightWristOffset = pose.RIGHT.WRIST.y - pose.RIGHT.HIP.y;

    // --- 1. CALIBRATION PHASE (30 frames) ---
    if (!this.calibrationData.isCalibrated) {
      this.calibrationData.framesCaptured++;

      // Accumulate neutral wrist position
      this.calibrationData.neutralWristOffset += (leftWristOffset + rightWristOffset) / 2;

      // Accumulate body measurements in pixels (average of left and right sides)
      const ankle = (pose.LEFT.ANKLE.y + pose.RIGHT.ANKLE.y) / 2;
      const nose = (pose.LEFT.NOSE.y + pose.RIGHT.NOSE.y) / 2;
      const shoulder = (pose.LEFT.SHOULDER.y + pose.RIGHT.SHOULDER.y) / 2;
      const hip = (pose.LEFT.HIP.y + pose.RIGHT.HIP.y) / 2;
      const knee = (pose.LEFT.KNEE.y + pose.RIGHT.KNEE.y) / 2;

      // Use left side for arm measurements
      const elbow = pose.LEFT.ELBOW.y;
      const wrist = pose.LEFT.WRIST.y;

      this.calibrationData.ankleToNosePixels += Math.abs(nose - ankle);
      this.calibrationData.shoulderToHipPixels += Math.abs(shoulder - hip);
      this.calibrationData.hipToKneePixels += Math.abs(hip - knee);
      this.calibrationData.kneeToAnklePixels += Math.abs(knee - ankle);
      this.calibrationData.shoulderToElbowPixels += Math.abs(shoulder - elbow);
      this.calibrationData.elbowToWristPixels += Math.abs(elbow - wrist);

      // Complete calibration after 30 frames
      if (this.calibrationData.framesCaptured >= 30) {
        // Average all measurements
        this.calibrationData.neutralWristOffset /= 30;
        this.calibrationData.ankleToNosePixels /= 30;
        this.calibrationData.shoulderToHipPixels /= 30;
        this.calibrationData.hipToKneePixels /= 30;
        this.calibrationData.kneeToAnklePixels /= 30;
        this.calibrationData.shoulderToElbowPixels /= 30;
        this.calibrationData.elbowToWristPixels /= 30;

        // Calculate pixel-to-cm conversion ratio
        const ankleToNoseCm = this.calibrationData.fullHeightCm - this.THRESHOLDS.NOSETOHEADCM;
        this.calibrationData.pixelToCmRatio = ankleToNoseCm / this.calibrationData.ankleToNosePixels;

        // Convert all measurements to cm
        this.calibrationData.torsoLengthCm = this.calibrationData.shoulderToHipPixels * this.calibrationData.pixelToCmRatio;
        this.calibrationData.thighLengthCm = this.calibrationData.hipToKneePixels * this.calibrationData.pixelToCmRatio;
        this.calibrationData.shinLengthCm = this.calibrationData.kneeToAnklePixels * this.calibrationData.pixelToCmRatio;
        this.calibrationData.upperArmLengthCm = this.calibrationData.shoulderToElbowPixels * this.calibrationData.pixelToCmRatio;
        this.calibrationData.forearmLengthCm = this.calibrationData.elbowToWristPixels * this.calibrationData.pixelToCmRatio;

        this.calibrationData.isCalibrated = true;

        console.log("=== CALIBRATION COMPLETE ===");
        console.log(`Height: ${this.calibrationData.userHeightInches}" (${this.calibrationData.fullHeightCm.toFixed(1)} cm)`);
        console.log(`Pixel-to-CM Ratio: ${this.calibrationData.pixelToCmRatio.toFixed(4)} cm/pixel`);
        console.log(`Torso: ${this.calibrationData.torsoLengthCm.toFixed(1)} cm`);
        console.log(`Thigh: ${this.calibrationData.thighLengthCm.toFixed(1)} cm`);
        console.log(`Shin: ${this.calibrationData.shinLengthCm.toFixed(1)} cm`);
        console.log(`Upper Arm: ${this.calibrationData.upperArmLengthCm.toFixed(1)} cm`);
        console.log(`Forearm: ${this.calibrationData.forearmLengthCm.toFixed(1)} cm`);
        console.log("============================");
      }

      // Show calibration progress
      this.drawCalibrationUI(ctx, canvas, pose, this.calibrationData.framesCaptured);
      return null;
    }

    // --- 2. FLEXIBLE UNLOCK (Hand Reset) ---
    const leftAtHome = Math.abs(leftWristOffset - this.calibrationData.neutralWristOffset) < 0.10;
    const rightAtHome = Math.abs(rightWristOffset - this.calibrationData.neutralWristOffset) < 0.10;
    const isTall = currentTorso > (this.calibrationData.shoulderToHipPixels * 0.85);

    if (leftAtHome && rightAtHome && isTall) {
      this.state.resetProgress++;
      this.drawResetUI(ctx, canvas, pose);

      if (this.state.resetProgress >= this.THRESHOLDS.RESETDURATIONFRAMES) {
        console.log("✅ Flexible Unlock Triggered");
        this.reset();
        return null;
      }
    } else {
      this.state.resetProgress = 0;
    }

    // --- 3. SIDE LOCKING ---
    if (this.state.lockedSide === "unknown") {
      if (Math.abs(pose.LEFT.WRIST.y - pose.RIGHT.WRIST.y) > 0.1) {
        this.state.lockedSide = pose.LEFT.WRIST.y < pose.RIGHT.WRIST.y ? "LEFT" : "RIGHT";
      } else {
        return null;
      }
    }

    const side = this.state.lockedSide;
    const wrist = pose[side].WRIST;
    const hip = pose[side].HIP;
    const shoulder = pose[side].SHOULDER;
    const nose = pose[side].NOSE;

    // --- 4. VELOCITY CALCULATION (using real torso length in cm) ---
    if (!this.state.calibration && shoulder && hip) {
      this.state.calibration = this.calibrationData.torsoLengthCm / 100; // Convert to meters
    }

    if (this.state.lastTimestamp > 0 && this.state.calibration) {
      const dy = ((wrist.y - this.state.lastWristY) * this.canvasHeight);
      const dt = (timestamp - this.state.lastTimestamp) / 1000;

      if (dt > 0 && dt < 0.1) {
        // Convert pixel displacement to meters using pixel-to-cm ratio
        const dyMeters = (dy * this.calibrationData.pixelToCmRatio) / 100;
        const vy = dyMeters / dt;
        this.state.smoothedVy = this.THRESHOLDS.VELOCITYALPHA * vy + (1 - this.THRESHOLDS.VELOCITYALPHA) * this.state.smoothedVy;
      }
    }

    this.state.lastTimestamp = timestamp;
    this.state.lastWristY = wrist.y;

    // --- 5. POSE DETECTION ---
    const hinged = Math.abs(shoulder.y - hip.y) < this.THRESHOLDS.HINGE;
    const atRack = Math.abs(wrist.y - shoulder.y) < 0.15 && 
                   Math.abs(wrist.x - (pose.LEFT.SHOULDER.x + pose.RIGHT.SHOULDER.x)/2) < 0.2;

    if (atRack && !hinged) {
      if (++this.state.rackFrameCount >= this.THRESHOLDS.RACKLOCKFRAMES) {
        this.state.currentPose = "RACK";
      }
    } else {
      this.state.rackFrameCount = 0;
    }

    if (wrist.y > pose[side].KNEE.y + this.THRESHOLDS.KNEECROSS) {
      this.state.currentPose = "HINGE";
    } else if (this.state.phase === "IDLE") {
      this.state.currentPose = "NONE";
    }

    // --- 6. MOVEMENT LOGIC ---
    let result = null;

    if (this.state.phase === "IDLE") {
      if (Math.abs(this.state.smoothedVy) > this.THRESHOLDS.PULLVELOCITYTRIGGER) {
        this.state.phase = "PULLING";
        this.state.movementStartPose = this.state.currentPose;
        this.state.currentRepPeak = 0;
        this.state.hipCrossedUpward = false;
      }
    } else if (this.state.phase === "PULLING") {
      this.state.currentRepPeak = Math.max(this.state.currentRepPeak, Math.abs(this.state.smoothedVy));

      if (wrist.y < hip.y) {
        this.state.hipCrossedUpward = true;
      }

      if (Math.abs(this.state.smoothedVy) < this.THRESHOLDS.LOCKOUTVYCUTOFF) {
        result = this.classify(this.state.movementStartPose, wrist, shoulder, hip, nose, hinged, this.state.hipCrossedUpward);
        if (result) {
          result.velocity = this.state.currentRepPeak;
          this.state.phase = "LOCKED";
        }
      }
    } else if (this.state.phase === "LOCKED") {
      if (wrist.y > hip.y || atRack) {
        this.state.phase = "IDLE";
      }
    }

    return result;
  }

  classify(start, w, s, h, nose, hinged, crossed) {
    const isOverhead = w.y < nose.y - 0.05;
    const isAtShoulder = w.y < s.y + 0.12 && w.y > s.y - 0.08;

    if (start === "HINGE") {
      if (isOverhead) return { type: "SNATCH" };
      if (isAtShoulder && !hinged && crossed) return { type: "CLEAN" };
      if (w.y < h.y && w.y > s.y && crossed) return { type: "SWING" };
    } else if (start === "RACK") {
      if (isOverhead) return { type: "PRESS" };
    }

    return null;
  }

  drawCalibrationUI(ctx, canvas, pose, frames) {
    const centerX = ((pose.LEFT.SHOULDER.x + pose.RIGHT.SHOULDER.x) / 2) * canvas.width;
    const centerY = ((pose.LEFT.SHOULDER.y + pose.LEFT.HIP.y) / 2) * canvas.height;
    const pct = frames / 30;

    // Background circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, 60, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.lineWidth = 10;
    ctx.stroke();

    // Progress arc
    ctx.beginPath();
    ctx.arc(centerX, centerY, 60, -Math.PI/2, -Math.PI/2 + (Math.PI * 2 * pct));
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 10;
    ctx.stroke();

    // Text
    ctx.fillStyle = "#fff";
    ctx.font = "bold 20px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("CALIBRATING", centerX, centerY - 10);
    ctx.font = "16px sans-serif";
    ctx.fillText(`${frames}/30`, centerX, centerY + 15);
  }

  drawResetUI(ctx, canvas, pose) {
    const centerX = ((pose.LEFT.SHOULDER.x + pose.RIGHT.SHOULDER.x) / 2) * canvas.width;
    const centerY = ((pose.LEFT.SHOULDER.y + pose.LEFT.HIP.y) / 2) * canvas.height;
    const pct = this.state.resetProgress / this.THRESHOLDS.RESETDURATIONFRAMES;

    ctx.beginPath();
    ctx.arc(centerX, centerY, 40, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.lineWidth = 8;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(centerX, centerY, 40, -Math.PI/2, -Math.PI/2 + (Math.PI * 2 * pct));
    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth = 8;
    ctx.stroke();
  }
}

// === APP CORE ===
const app = {
  video: null,
  canvas: null,
  ctx: null,
  landmarker: null,
  stateMachine: null,
  isModelLoaded: false,
  isTestRunning: false,
  totalReps: 0,
  lastMove: "READY",
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

  const saveBtn = document.getElementById("btn-save");
  if (saveBtn) saveBtn.onclick = exportToMake;

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
    app.canvas.width = app.video.videoWidth;
    app.canvas.height = app.video.videoHeight;

    // Get height from input and initialize state machine
    const heightInput = document.getElementById("height-input");
    const heightInches = parseFloat(heightInput.value) || 68; // Default 68" if not set

    app.stateMachine = new VBTStateMachine(app.canvas.height);
    app.stateMachine.setUserHeight(heightInches);

    document.getElementById("btn-start-test").disabled = false;
  };
}

async function startCamera() {
  const s = await navigator.mediaDevices.getUserMedia({ video: true });
  app.video.srcObject = s;
  app.video.onloadedmetadata = () => {
    app.canvas.width = app.video.videoWidth;
    app.canvas.height = app.video.videoHeight;

    const heightInput = document.getElementById("height-input");
    const heightInches = parseFloat(heightInput.value) || 68;

    app.stateMachine = new VBTStateMachine(app.canvas.height);
    app.stateMachine.setUserHeight(heightInches);

    document.getElementById("btn-start-test").disabled = false;
  };
}

function toggleTest() {
  app.isTestRunning = !app.isTestRunning;
  document.getElementById("btn-start-test").innerText = app.isTestRunning ? "PAUSE" : "START";
  if (app.isTestRunning) {
    app.video.play();
  } else {
    app.video.pause();
  }
}

async function masterLoop(ts) {
  requestAnimationFrame(masterLoop);
  if (!app.isModelLoaded || !app.video.readyState) return;

  app.ctx.drawImage(app.video, 0, 0, app.canvas.width, app.canvas.height);

  const results = app.landmarker.detectForVideo(app.video, ts);
  if (results?.landmarks?.length > 0) {
    const raw = results.landmarks[0];
    const pose = {
      LEFT: { WRIST: raw[15], SHOULDER: raw[11], HIP: raw[23], KNEE: raw[25], ANKLE: raw[27], ELBOW: raw[13], NOSE: raw[0] },
      RIGHT: { WRIST: raw[16], SHOULDER: raw[12], HIP: raw[24], KNEE: raw[26], ANKLE: raw[28], ELBOW: raw[14], NOSE: raw[0] }
    };

    if (app.isTestRunning && app.stateMachine) {
      const move = app.stateMachine.update(pose, ts, app.ctx, app.canvas);
      if (move) record(move);
      drawUI(app.stateMachine.state, pose);
    }
  }
}

function record(m) {
  app.totalReps++;
  app.lastMove = m.type;
  app.history[m.type].push(m.velocity);

  const countEl = document.getElementById(`val-${m.type.toLowerCase()}s`);
  if (countEl) countEl.innerText = app.history[m.type].length;

  const moveEl = document.getElementById("detected-movement");
  if (moveEl) moveEl.innerText = m.type;
}

function resetSession() {
  app.totalReps = 0;
  app.lastMove = "READY";
  app.history = { CLEAN: [], PRESS: [], SNATCH: [], SWING: [] };

  if (app.video.src) {
    app.video.pause();
    app.video.currentTime = 0;
  }

  const countEls = ["val-cleans", "val-presses", "val-snatches", "val-swings", "val-total-reps"];
  countEls.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = "0";
  });

  const velEls = ["val-clean-velocity", "val-press-velocity", "val-snatch-velocity", "val-swing-velocity", "val-velocity"];
  velEls.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = "0.00";
  });

  console.log("Session Reset");
}

function drawUI(s, p) {
  document.getElementById("val-velocity").innerText = Math.abs(s.smoothedVy).toFixed(2);

  app.ctx.save();
  app.ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
  app.ctx.fillRect(20, 20, 180, 90);

  app.ctx.fillStyle = "#ffffff";
  app.ctx.font = "bold 40px sans-serif";
  app.ctx.fillText(app.totalReps, 40, 65);

  app.ctx.font = "bold 14px sans-serif";
  app.ctx.fillText("TOTAL REPS", 40, 85);

  app.ctx.fillStyle = "#3b82f6";
  app.ctx.font = "bold 18px sans-serif";
  app.ctx.fillText(app.lastMove, 100, 65);
  app.ctx.restore();

  if (s.lockedSide !== "unknown") {
    const w = p[s.lockedSide].WRIST;
    app.ctx.fillStyle = "#10b981";
    app.ctx.beginPath();
    app.ctx.arc(w.x * app.canvas.width, w.y * app.canvas.height, 12, 0, Math.PI * 2);
    app.ctx.fill();
  }
}

function exportToMake() {
  console.log("Export functionality placeholder");
}

initializeApp();
