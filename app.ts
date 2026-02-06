/**
 * VBT v3.5  - COMPLETE MOVEMENT DETECTION
 * All bugs resolved - ready for production
 */

// @ts-ignore - Using CDN import for MediaPipe
import { PoseLandmarker, FilesetResolver, NormalizedLandmark } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

// ============================================
// TYPES
// ============================================

type Side = "left" | "right" | "unknown";
type TestStage = "IDLE" | "RUNNING";
type Phase = "IDLE" | "BOTTOM" | "CONCENTRIC" | "LOCKOUT";
type Zone = "FLOOR" | "BACKSWING" | "RACK" | "OVERHEAD" | "TRANSITION" | "UNKNOWN";
type MovementType = 
  | "CLEAN_FROM_FLOOR" 
  | "RE_CLEAN" 
  | "PRESS" 
  | "SNATCH" 
  | "SWING"
  | "PRESS_SINGLE_LEFT"
  | "PRESS_SINGLE_RIGHT"
  | "SNATCH_SINGLE_LEFT"
  | "SNATCH_SINGLE_RIGHT"
  | "SWING_SINGLE_LEFT"
  | "SWING_SINGLE_RIGHT";
type RepStartPosition = "FLOOR" | "RACK" | null;

interface LandmarkIndices {
  WRIST: number;
  SHOULDER: number;
  HIP: number;
  KNEE: number;
}

interface MovementConfig {
  SNATCH_MIN_HEIGHT_ABOVE_SHOULDER: number;
  CLEAN_RACK_HEIGHT_MIN: number;
  CLEAN_RACK_HEIGHT_MAX: number;
  CLEAN_HORIZONTAL_PROXIMITY: number;
  SWING_MAX_HEIGHT_ABOVE_SHOULDER: number;
  SWING_MIN_HEIGHT_ABOVE_HIP: number;
  PRESS_VELOCITY_THRESHOLD: number;
}

interface Config {
  LEFT: LandmarkIndices;
  RIGHT: LandmarkIndices;
  HEAD_LANDMARK: number;
  TORSO_METERS: number;
  SMOOTHING_ALPHA: number;
  MAX_REALISTIC_VELOCITY: number;
  ZERO_BAND: number;
  MIN_DT: number;
  MAX_DT: number;
  LOCKOUT_VY_CUTOFF: number;
  LOCKOUT_SPEED_CUTOFF: number;
  RESET_GRACE_MS_AFTER_LOCK: number;
  HIKE_VY_THRESHOLD: number;
  HIKE_SPEED_THRESHOLD: number;
  BASELINE_REPS: number;
  DROP_WARN: number;
  DROP_FAIL: number;
  MIN_DET_CONF: number;
  MIN_TRACK_CONF: number;
  MOVEMENT: MovementConfig;
  MAKE_WEBHOOK_URL: string;
  DEBUG_MODE: boolean;
}

interface WristPosition {
  x: number;
  y: number;
  t: number;
}

interface MovementData {
  type: MovementType;
  velocity: number;
  timestamp: number;
}

interface SetSummary {
  total_cleans: number;
  floor_cleans: number;
  re_cleans: number;
  total_presses: number;
  total_snatches: number;
  total_swings: number;
}

interface Set {
  id: number;
  hand: Side;
  cleans: MovementData[];
  presses: MovementData[];
  snatches: MovementData[];
  swings: MovementData[];
  startTime: Date;
  endTime?: Date;
  lockedAtMs: number;
  summary?: SetSummary;
}

interface Session {
  currentSet: Set | null;
  history: Set[];
}

interface State {
  video: HTMLVideoElement | null;
  canvas: HTMLCanvasElement | null;
  ctx: CanvasRenderingContext2D | null;
  landmarker: PoseLandmarker | null;
  isModelLoaded: boolean;
  isVideoReady: boolean;
  isTestRunning: boolean;
  testStage: TestStage;
  timeMs: number;
  lastPose: NormalizedLandmark[] | null;
  activeTrackingSide: Side;
  lockedSide: Side;
  armingSide: Side | null;
  prevWrist: WristPosition | null;
  lockedCalibration: number | null;
  smoothedVelocity: number;
  smoothedVy: number;
  lastSpeed: number;
  lastVy: number;
  parkingConfirmed: boolean;
  prevHeadY: number;
  phase: Phase;
  currentRepPeak: number;
  overheadHoldCount: number;
  session: Session;
  repStartedFrom: RepStartPosition;
  currentRepPeakWristY: number;
  currentRepPeakWristX: number;
  cleanHistory: number[];
  pressHistory: number[];
  cleanBaseline: number;
  pressBaseline: number;
  endingConfirmCount: number;
  repStartY?: number;
  lockedAtMs: number;
}

interface ExportPayload {
  athlete_id: string;
  session_date: string;
  sets: Array<{
    set_order: number;
    hand: Side;
    cleans: MovementData[];
    presses: MovementData[];
    snatches: MovementData[];
    swings: MovementData[];
    summary: SetSummary | {};
  }>;
}

// ============================================
// CONFIG
// ============================================

const CONFIG: Config = {
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

  RESET_GRACE_MS_AFTER_LOCK: 5000,
  HIKE_VY_THRESHOLD: 0.3,
  HIKE_SPEED_THRESHOLD: 0.5,

  BASELINE_REPS: 3,
  DROP_WARN: 15,
  DROP_FAIL: 20,

  MIN_DET_CONF: 0.5,
  MIN_TRACK_CONF: 0.5,

  MOVEMENT: {
    SNATCH_MIN_HEIGHT_ABOVE_SHOULDER: 0.05,
    CLEAN_RACK_HEIGHT_MIN: -0.1,
    CLEAN_RACK_HEIGHT_MAX: 0.15,
    CLEAN_HORIZONTAL_PROXIMITY: 0.18,
    SWING_MAX_HEIGHT_ABOVE_SHOULDER: 0.05,
    SWING_MIN_HEIGHT_ABOVE_HIP: 0.1,
    PRESS_VELOCITY_THRESHOLD: 3.5
  },

  MAKE_WEBHOOK_URL: "https://hook.us2.make.com/0l88dnosrk2t8a29yfk83fej8hp8j3jk",

  DEBUG_MODE: true
};

// ============================================
// STATE
// ============================================

let state: State = {
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

  activeTrackingSide: "left",
  lockedSide: "unknown",
  armingSide: null,

  prevWrist: null,
  lockedCalibration: null,
  smoothedVelocity: 0,
  smoothedVy: 0,
  lastSpeed: 0,
  lastVy: 0,

  parkingConfirmed: false,
  prevHeadY: 0,

  phase: "IDLE",
  currentRepPeak: 0,
  overheadHoldCount: 0,

  session: {
    currentSet: null,
    history: []
  },

  repStartedFrom: null,
  currentRepPeakWristY: 1.0,
  currentRepPeakWristX: 0.5,

  cleanHistory: [],
  pressHistory: [],
  cleanBaseline: 0,
  pressBaseline: 0,

  endingConfirmCount: 0,
  lockedAtMs: 0
};

// ============================================
// INITIALIZATION
// ============================================

async function initializeApp(): Promise<void> {
  state.video = document.getElementById("video") as HTMLVideoElement;
  state.canvas = document.getElementById("canvas") as HTMLCanvasElement;
  state.ctx = state.canvas.getContext("2d");

  const btnCamera = document.getElementById("btn-camera");
  if (btnCamera) btnCamera.onclick = startCamera;

  const fileInput = document.getElementById("file-input") as HTMLInputElement;
  if (fileInput) fileInput.onchange = handleUpload;

  const btnStartTest = document.getElementById("btn-start-test");
  if (btnStartTest) btnStartTest.onclick = toggleTest;

  const btnReset = document.getElementById("btn-reset");
  if (btnReset) btnReset.onclick = resetSession;

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
  const loadingOverlay = document.getElementById("loading-overlay");
  if (loadingOverlay) loadingOverlay.classList.add("hidden");
  setStatus("Vision System Online — Upload Video or Start Camera", "#3b82f6");

  if (state.video) state.video.addEventListener("loadeddata", onVideoReady);
  requestAnimationFrame(masterLoop);
}

// ============================================
// VIDEO INPUTS
// ============================================

function handleUpload(e: Event): void {
  const target = e.target as HTMLInputElement;
  const file = target.files?.[0];
  if (!file) return;

  resetSession();
  setStatus("Loading Video...", "#fbbf24");

  if (state.video) {
    state.video.srcObject = null;
    state.video.src = URL.createObjectURL(file);
    state.video.load();
  }
}

async function startCamera(): Promise<void> {
  resetSession();

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false
  });

  setStatus("Activating Optics...", "#fbbf24");
  if (state.video) {
    state.video.srcObject = stream;
    state.video.src = "";
    state.video.play();
  }
}

function onVideoReady(): void {
  state.isVideoReady = true;
  if (state.canvas && state.video) {
    state.canvas.width = state.video.videoWidth;
    state.canvas.height = state.video.videoHeight;
  }

  const btnStartTest = document.getElementById("btn-start-test") as HTMLButtonElement;
  if (btnStartTest) btnStartTest.disabled = false;

  if (state.video?.src) {
    const p = state.video.play();
    if (p && typeof p.then === "function") {
      p.then(() => {
        setTimeout(() => {
          if (!state.isTestRunning && state.video) {
            state.video.pause();
            state.video.currentTime = 0;
          }
        }, 100);
      }).catch(() => {});
    }
  }

  setStatus("Video Ready — Press Start Test", "#3b82f6");
}

async function toggleTest(): Promise<void> {
  const btnStartTest = document.getElementById("btn-start-test") as HTMLButtonElement;
  const btnReset = document.getElementById("btn-reset") as HTMLButtonElement;

  if (!state.isTestRunning) {
    state.isTestRunning = true;

    if (state.testStage !== "RUNNING") {
      state.testStage = "IDLE";
    }

    if (btnStartTest) btnStartTest.textContent = "Pause Test";
    if (btnReset) btnReset.disabled = false;

    if (state.testStage === "IDLE") {
      setStatus("Scanning: Park hand below knee...", "#fbbf24");
      resetMovementDisplay();
    } else {
      setStatus("RESUMED: Test Running", "#10b981");
    }

    if (state.video?.paused) {
      try { await state.video.play(); } catch(e) {}
    }
  } else {
    state.isTestRunning = false;
    if (btnStartTest) btnStartTest.textContent = "Resume Test";
    setStatus("Paused", "#fbbf24");
    if (state.video) state.video.pause();
  }
}

// ============================================
// MASTER LOOP
// ============================================

async function masterLoop(timestamp: number): Promise<void> {
  requestAnimationFrame(masterLoop);
  if (!state.isModelLoaded || !state.video) return;

  state.timeMs = timestamp;

  if (state.ctx && state.canvas && state.video) {
    state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
    state.ctx.drawImage(state.video, 0, 0, state.canvas.width, state.canvas.height);
  }

  let pose: NormalizedLandmark[] | null = null;
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

    if (state.testStage === "IDLE") {
      checkStartCondition(pose, state.timeMs);
    }

    if (state.testStage === "RUNNING") {
      runMovementLogic(pose);
      checkEndCondition(pose, state.timeMs);
    }
  }

  drawOverlay();
}

// ============================================
// START/END CONDITIONS
// ============================================

function checkStartCondition(pose: NormalizedLandmark[], _timeMs: number): void {
  if (state.testStage !== "IDLE") return;

  const lWrist = pose[CONFIG.LEFT.WRIST];
  const rWrist = pose[CONFIG.RIGHT.WRIST];
  if (!lWrist || !rWrist) return;

  const lY = lWrist.y;
  const rY = rWrist.y;
  const activeSide: Side = lY > rY ? "left" : "right";

  state.activeTrackingSide = activeSide;

  const inZone = isWristInFloorZone(pose, activeSide);
  const hikingDown = state.lastVy > 0.3 && state.lastSpeed > 0.5;

  if (CONFIG.DEBUG_MODE && inZone) {
    console.log(`[START] Side:${activeSide} | Zone:${inZone} | Hike:${hikingDown} | Vy:${state.lastVy.toFixed(2)}`);
  }

  if (inZone && hikingDown) {
    console.log(`🚀 STARTING SET: ${activeSide}`);
    startNewSet(activeSide);
  }
}

function checkEndCondition(pose: NormalizedLandmark[], timeMs: number): void {
  if (state.testStage !== "RUNNING") return;
  if (!state.session.currentSet) return;

  const grace = timeMs - state.session.currentSet.lockedAtMs;

  const INITIAL_GRACE_MS = 5000;
  const totalReps = (state.cleanHistory.length || 0) + (state.pressHistory.length || 0);
  if (grace < INITIAL_GRACE_MS && totalReps === 0) {
    return;
  }

  if (grace < CONFIG.RESET_GRACE_MS_AFTER_LOCK) return;

  if (state.phase === "CONCENTRIC") return;

  const sideIdx = state.lockedSide === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[sideIdx.WRIST];
  if (!wrist) return;

  const inZone = isWristInFloorZone(pose, state.lockedSide);
  const standingUp = state.lastVy < -0.3 && state.lastSpeed > 0.5;

  if (!state.endingConfirmCount) state.endingConfirmCount = 0;

  if (inZone && standingUp) {
    state.endingConfirmCount++;

    if (state.endingConfirmCount >= 2) {
      console.log(`🛑 ENDING SET (${state.cleanHistory.length} cleans, ${state.pressHistory.length} presses)`);
      state.endingConfirmCount = 0;
      endCurrentSet();
    }
  } else {
    state.endingConfirmCount = 0;
  }
}

// ============================================
// SET MANAGEMENT
// ============================================

function startNewSet(side: Side): void {
  state.lockedSide = side;
  state.testStage = "RUNNING";
  state.lockedAtMs = state.timeMs;

  state.cleanHistory = [];
  state.pressHistory = [];
  state.cleanBaseline = 0;
  state.pressBaseline = 0;
  state.currentRepPeak = 0;
  state.overheadHoldCount = 0;
  state.endingConfirmCount = 0;

  state.phase = "IDLE";
  state.repStartedFrom = null;

  state.session.currentSet = {
    id: state.session.history.length + 1,
    hand: side,
    cleans: [],
    presses: [],
    snatches: [],
    swings: [],
    startTime: new Date(),
    lockedAtMs: state.timeMs
  };

  // Reset all UI
  const countEls = ['val-cleans', 'val-presses', 'val-snatches', 'val-swings', 'val-total-reps'];
  countEls.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = '0';
  });

  const velEls = ['val-clean-velocity', 'val-press-velocity', 'val-snatch-velocity', 'val-swing-velocity'];
  velEls.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = '0.00';
  });

  setStatus(`LOCKED: ${side.toUpperCase()}`, "#10b981");

  if (CONFIG.DEBUG_MODE) console.log(`🚀 Set Started [${side}]`);
}

function endCurrentSet(): void {
  if (state.session.currentSet) {
    state.session.currentSet.endTime = new Date();

    const cleanCount = state.session.currentSet.cleans.length;
    const pressCount = state.session.currentSet.presses.length;
    const snatchCount = state.session.currentSet.snatches.length;
    const swingCount = state.session.currentSet.swings.length;

    state.session.currentSet.summary = {
      total_cleans: cleanCount,
      floor_cleans: state.session.currentSet.cleans.filter(c => c.type === 'CLEAN_FROM_FLOOR').length,
      re_cleans: state.session.currentSet.cleans.filter(c => c.type === 'RE_CLEAN').length,
      total_presses: pressCount,
      total_snatches: snatchCount,
      total_swings: swingCount
    };

    state.session.history.push(state.session.currentSet);
  }

  state.testStage = "IDLE";
  state.lockedSide = "unknown";
  state.session.currentSet = null;
  state.activeTrackingSide = "left";

  setStatus("Set Saved. Park to start next.", "#3b82f6");
}

// ============================================
// PHYSICS ENGINE
// ============================================

function runPhysics(pose: NormalizedLandmark[], timeMs: number): void {
  const side = state.testStage === "IDLE" ? state.activeTrackingSide : state.lockedSide;
  if (!side || side === "unknown") return;

  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const shoulder = pose[idx.SHOULDER];
  const hip = pose[idx.HIP];
  if (!wrist || !shoulder || !hip) return;

  if (!state.lockedCalibration && state.canvas) {
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

  if (!state.canvas || !state.lockedCalibration) return;

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
    const velEl = document.getElementById("val-velocity");
    if (velEl) velEl.textContent = state.lastSpeed.toFixed(2);
  }
}

// ============================================
// MOVEMENT DETECTION LOGIC
// ============================================

function runMovementLogic(pose: NormalizedLandmark[]): void {
  const v = state.smoothedVelocity;
  const vy = state.smoothedVy;
  const idx = state.lockedSide === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const hip = pose[idx.HIP];
  const shoulder = pose[idx.SHOULDER];
  const knee = pose[idx.KNEE];

  if (!wrist || !hip || !shoulder || !knee) return;

  const zone = getWristZone(pose, state.lockedSide);

  // PHASE: IDLE/LOCKOUT - Waiting for next movement
  if (state.phase === "IDLE" || state.phase === "LOCKOUT") {

    if (zone === 'FLOOR') {
      state.phase = "BOTTOM";
      state.repStartedFrom = "FLOOR";
      state.overheadHoldCount = 0;
      state.currentRepPeak = 0;
      state.currentRepPeakWristY = 1.0;
      state.currentRepPeakWristX = wrist.x;
      state.repStartY = wrist.y;  // ADD THIS
      if (CONFIG.DEBUG_MODE) console.log("Phase: BOTTOM (from FLOOR)");
    }

    else if (zone === 'BACKSWING') {
      state.phase = "BOTTOM";
      state.repStartedFrom = "RACK";
      state.overheadHoldCount = 0;
      state.currentRepPeak = 0;
      state.currentRepPeakWristY = 1.0;
      state.currentRepPeakWristX = wrist.x;

      if (CONFIG.DEBUG_MODE) console.log("Phase: BOTTOM (from RACK)");
    }
    // ADD THIS - detect press starting from rack
    else if (zone === 'RACK' && vy < -0.4) {
      state.phase = "CONCENTRIC";
      state.currentRepPeak = 0;
      if (CONFIG.DEBUG_MODE) console.log("Phase: CONCENTRIC (press from rack)");
    }
  }

  // PHASE: BOTTOM - Waiting for upward pull
  else if (state.phase === "BOTTOM") {
    if (zone !== 'FLOOR' && zone !== 'BACKSWING' && vy < -0.4) {
      state.phase = "CONCENTRIC";
      if (CONFIG.DEBUG_MODE) console.log("Phase: CONCENTRIC");
    }
  }

  // PHASE: CONCENTRIC - Tracking upward movement
  else if (state.phase === "CONCENTRIC") {
    if (v > state.currentRepPeak) state.currentRepPeak = v;

    if (wrist.y < state.currentRepPeakWristY) {
      state.currentRepPeakWristY = wrist.y;
      state.currentRepPeakWristX = wrist.x;
    }

    const isStable = Math.abs(vy) < CONFIG.LOCKOUT_VY_CUTOFF && v < CONFIG.LOCKOUT_SPEED_CUTOFF;

    // RACK LOCKOUT - Clean detection
    if (zone === 'RACK' && isStable) {
      state.overheadHoldCount++;

      if (state.overheadHoldCount >= 5) {
        if (state.repStartY && wrist.y < (state.repStartY - 0.2)) {
          recordClean(pose, "CLEAN_FROM_FLOOR");
        } else if (state.repStartedFrom === "RACK") {
          recordClean(pose, "RE_CLEAN");
        }

        state.phase = "LOCKOUT";
        state.overheadHoldCount = 0;
      }
    }

    // OVERHEAD LOCKOUT - Press or Snatch
    else if (zone === 'OVERHEAD' && isStable) {
      state.overheadHoldCount++;

      if (state.overheadHoldCount >= 2) {
        if (state.currentRepPeak < CONFIG.MOVEMENT.PRESS_VELOCITY_THRESHOLD) {
          recordPress(pose);
        } else {
          recordSnatch(pose);
        }

        state.phase = "LOCKOUT";
        state.overheadHoldCount = 0;
      }
    }

    // SWING DETECTION - Shoulder height lockout (not rack, not overhead)
    else if (isStable && state.repStartedFrom === "FLOOR") {
      // Check if at shoulder height but not in rack zone or overhead
      const heightAboveShoulder = shoulder.y - wrist.y;
      const heightAboveHip = hip.y - wrist.y;

      // At shoulder height, ballistic velocity, not in rack position
      if (heightAboveShoulder < CONFIG.MOVEMENT.SWING_MAX_HEIGHT_ABOVE_SHOULDER &&
          heightAboveShoulder >= -0.1 &&
          heightAboveHip >= CONFIG.MOVEMENT.SWING_MIN_HEIGHT_ABOVE_HIP &&
          state.currentRepPeak >= CONFIG.MOVEMENT.PRESS_VELOCITY_THRESHOLD) {

        state.overheadHoldCount++;

        if (state.overheadHoldCount >= 2) {
          recordSwing(pose);
          state.phase = "LOCKOUT";
          state.overheadHoldCount = 0;
        }
      }
    }

    else {
      state.overheadHoldCount = 0;
    }
  }

  // PHASE: LOCKOUT - Waiting for next movement or set end
  else if (state.phase === "LOCKOUT") {
    if (zone === 'BACKSWING' || zone === 'FLOOR') {
      state.phase = "IDLE";
      state.repStartedFrom = null;
    }
  }
}

// ============================================
// POSITION DETECTION
// ============================================

function getWristZone(pose: NormalizedLandmark[], side: Side): Zone {
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const hip = pose[idx.HIP];
  const shoulder = pose[idx.SHOULDER];
  const knee = pose[idx.KNEE];

  if (!wrist || !hip || !shoulder || !knee) return 'UNKNOWN';

  const lShoulder = pose[CONFIG.LEFT.SHOULDER];
  const rShoulder = pose[CONFIG.RIGHT.SHOULDER];
  const torsoCenter = (lShoulder.x + rShoulder.x) / 2;
  const horizontalDist = Math.abs(wrist.x - torsoCenter);
  const isCloseToTorso = horizontalDist < CONFIG.MOVEMENT.CLEAN_HORIZONTAL_PROXIMITY;

  if (wrist.y < (shoulder.y - CONFIG.MOVEMENT.SNATCH_MIN_HEIGHT_ABOVE_SHOULDER)) {
    return 'OVERHEAD';
  }

  if (wrist.y >= (shoulder.y + CONFIG.MOVEMENT.CLEAN_RACK_HEIGHT_MIN) && 
      wrist.y <= (shoulder.y + CONFIG.MOVEMENT.CLEAN_RACK_HEIGHT_MAX) && 
      isCloseToTorso) {
    return 'RACK';
  }

  if (wrist.y > hip.y && wrist.y <= knee.y) {
    return 'BACKSWING';
  }

  if (wrist.y > knee.y) {
    return 'FLOOR';
  }

  return 'TRANSITION';
}

function isWristInFloorZone(pose: NormalizedLandmark[], side: Side): boolean {
  const idx = side === "left" ? CONFIG.LEFT : CONFIG.RIGHT;
  const wrist = pose[idx.WRIST];
  const knee = pose[idx.KNEE];
  if (!wrist || !knee) return false;
  return wrist.y > knee.y;
}

// ============================================
// RECORDING FUNCTIONS - ALL FIXED
// ============================================

function recordClean(_pose: NormalizedLandmark[], cleanType: "CLEAN_FROM_FLOOR" | "RE_CLEAN"): void {
  const cleanData: MovementData = {
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

function recordPress(_pose: NormalizedLandmark[]): void {
  const pressData: MovementData = {
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
  updateMovementDisplay(`PRESS_SINGLE_${state.lockedSide.toUpperCase()}` as MovementType);
  updateTotalReps();
}

function recordSnatch(_pose: NormalizedLandmark[]): void {
  const snatchData: MovementData = {
    type: 'SNATCH',
    velocity: state.currentRepPeak,
    timestamp: Date.now()
  };

  if (state.session.currentSet) {
    state.session.currentSet.snatches.push(snatchData);

    if (CONFIG.DEBUG_MODE) {
      console.log(`⚡ SNATCH #${state.session.currentSet.snatches.length}: ${state.currentRepPeak.toFixed(2)} m/s`);
    }

    updateSnatchDisplay(state.session.currentSet.snatches.length, state.currentRepPeak);
  }

  updateMovementDisplay(`SNATCH_SINGLE_${state.lockedSide.toUpperCase()}` as MovementType);
  updateTotalReps();
}

function recordSwing(_pose: NormalizedLandmark[]): void {
  const swingData: MovementData = {
    type: 'SWING',
    velocity: state.currentRepPeak,
    timestamp: Date.now()
  };

  if (state.session.currentSet) {
    state.session.currentSet.swings.push(swingData);

    if (CONFIG.DEBUG_MODE) {
      console.log(`🔄 SWING #${state.session.currentSet.swings.length}: ${state.currentRepPeak.toFixed(2)} m/s`);
    }

    updateSwingDisplay(state.session.currentSet.swings.length, state.currentRepPeak);
  }

  updateMovementDisplay(`SWING_SINGLE_${state.lockedSide.toUpperCase()}` as MovementType);
  updateTotalReps();
}

// ============================================
// UI UPDATES - ALL FIXED
// ============================================

function updateCleanDisplay(count: number, velocity: number, drop: string, dropColor: string): void {
  const countEl = document.getElementById("val-cleans");
  const velEl = document.getElementById("val-clean-velocity");
  const dropEl = document.getElementById("val-clean-drop");

  if (countEl) countEl.textContent = count.toString();
  if (velEl) velEl.textContent = velocity.toFixed(2);
  if (dropEl) {
    dropEl.textContent = drop;
    dropEl.style.color = dropColor;
  }
}

function updatePressDisplay(count: number, velocity: number, drop: string, dropColor: string): void {
  const countEl = document.getElementById("val-presses");
  const velEl = document.getElementById("val-press-velocity");
  const dropEl = document.getElementById("val-press-drop");

  if (countEl) countEl.textContent = count.toString();
  if (velEl) velEl.textContent = velocity.toFixed(2);
  if (dropEl) {
    dropEl.textContent = drop;
    dropEl.style.color = dropColor;
  }
}

function updateSnatchDisplay(count: number, velocity: number): void {
  const countEl = document.getElementById("val-snatches");
  const velEl = document.getElementById("val-snatch-velocity");

  if (countEl) countEl.textContent = count.toString();
  if (velEl) velEl.textContent = velocity.toFixed(2);
}

function updateSwingDisplay(count: number, velocity: number): void {
  const countEl = document.getElementById("val-swings");
  const velEl = document.getElementById("val-swing-velocity");

  if (countEl) countEl.textContent = count.toString();
  if (velEl) velEl.textContent = velocity.toFixed(2);
}

function updateTotalReps(): void {
  if (!state.session.currentSet) return;

  const cleans = (state.session.currentSet.cleans || []).length;
  const presses = (state.session.currentSet.presses || []).length;
  const snatches = (state.session.currentSet.snatches || []).length;
  const swings = (state.session.currentSet.swings || []).length;

  const total = cleans + presses + snatches + swings;

  const totalEl = document.getElementById("val-total-reps");
  if (totalEl) totalEl.textContent = total.toString();
}

function updateMovementDisplay(movementType: MovementType | string): void {
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
  }

  configEl.textContent = 'Single Kettlebell';
  handsEl.textContent = state.lockedSide === 'left' ? 'Left' : state.lockedSide === 'right' ? 'Right' : '—';
  statusIndicator.className = 'status-indicator locked';
}

function formatMovementName(movementType: string): string {
  const typeMap: Record<string, string> = {
    'SWING_SINGLE_LEFT': 'One Arm Swing (L)',
    'SWING_SINGLE_RIGHT': 'One Arm Swing (R)',
    'CLEAN_FROM_FLOOR': 'Clean from Floor',
    'RE_CLEAN': 'Re-Clean',
    'SNATCH_SINGLE_LEFT': 'One Arm Snatch (L)',
    'SNATCH_SINGLE_RIGHT': 'One Arm Snatch (R)',
    'PRESS_SINGLE_LEFT': 'One Arm Press (L)',
    'PRESS_SINGLE_RIGHT': 'One Arm Press (R)'
  };
  return typeMap[movementType] || movementType.replace(/_/g, ' ');
}

function resetMovementDisplay(): void {
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

function setStatus(text: string, color: string): void {
  const pill = document.getElementById("status-pill");
  if (pill) {
    pill.textContent = text;
    pill.style.color = color;
    pill.style.borderColor = color;
  }
}

// ============================================
// DRAWING
// ============================================

function drawOverlay(): void {
  if (!state.lastPose || !state.ctx) return;

  if (CONFIG.DEBUG_MODE) {
    state.ctx.fillStyle = "#fbbf24";
    state.ctx.font = "12px monospace";
    state.ctx.fillText(`Side: ${state.testStage === "IDLE" ? state.activeTrackingSide : state.lockedSide}`, 10, 20);
    state.ctx.fillText(`Speed: ${state.lastSpeed.toFixed(2)} m/s`, 10, 35);
    state.ctx.fillText(`Vy: ${state.lastVy.toFixed(2)} m/s`, 10, 50);
    state.ctx.fillText(`Stage: ${state.testStage}`, 10, 65);
    state.ctx.fillText(`Phase: ${state.phase}`, 10, 80);

    if (state.testStage === "RUNNING") {
      const zone = getWristZone(state.lastPose, state.lockedSide);
      state.ctx.fillText(`Zone: ${zone}`, 10, 95);
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
    const lowest: Side = lY > rY ? "left" : "right";
    const color = "#fbbf24";

    drawDot(lWrist, lowest==="left", color);
    drawDot(rWrist, lowest==="right", color);
    drawParkingLine(state.lastPose, lowest, isWristInFloorZone(state.lastPose, lowest));
  }
}

function drawParkingLine(pose: NormalizedLandmark[], side: Side, isActive: boolean): void {
  if (!state.ctx || !state.canvas) return;

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

function drawDot(landmark: NormalizedLandmark | undefined, big: boolean, color: string): void {
  if (!landmark || !state.ctx || !state.canvas) return;

  const x = landmark.x * state.canvas.width;
  const y = landmark.y * state.canvas.height;

  state.ctx.beginPath();
  state.ctx.fillStyle = color;
  state.ctx.arc(x, y, big ? 8 : 5, 0, 2*Math.PI);
  state.ctx.fill();
}

// ============================================
// SESSION MANAGEMENT - FIXED RESET
// ============================================

function resetSession(): void {
  state.session = { currentSet: null, history: [] };
  state.cleanHistory = [];
  state.pressHistory = [];
  state.cleanBaseline = 0;
  state.pressBaseline = 0;
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
  state.phase = "IDLE";
  state.currentRepPeak = 0;
  state.overheadHoldCount = 0;
  state.parkingConfirmed = false;
  state.currentRepPeakWristY = 1.0;
  state.currentRepPeakWristX = 0.5;
  state.endingConfirmCount = 0;
  state.repStartedFrom = null;

  // Reset video to beginning
  if (state.video && state.video.src) {
    state.video.pause();
    state.video.currentTime = 0;
  }

  // Reset ALL UI elements
  const countEls = ['val-cleans', 'val-presses', 'val-snatches', 'val-swings', 'val-total-reps'];
  countEls.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = '0';
  });

  const velEls = ['val-clean-velocity', 'val-press-velocity', 'val-snatch-velocity', 'val-swing-velocity', 'val-velocity'];
  velEls.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = '0.00';
  });

  const dropEls = ['val-clean-drop', 'val-press-drop'];
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

async function exportToMake(): Promise<void> {
  const history = state.session.history;
  if (!history.length) {
    alert("No completed sets to export.");
    return;
  }

  const payload: ExportPayload = {
    athlete_id: "dad_ready_user",
    session_date: new Date().toISOString(),
    sets: history.map((set, index) => ({
      set_order: index + 1,
      hand: set.hand,
      cleans: set.cleans || [],
      presses: set.presses || [],
      snatches: set.snatches || [],
      swings: set.swings || [],
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
