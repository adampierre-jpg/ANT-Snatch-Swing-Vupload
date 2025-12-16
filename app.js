import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const CONFIG = {
    WRIST_ID: 16,
    SHOULDER_ID: 12,
    HIP_ID: 24,
    SMOOTHING: 3,
    DROP_WARN: 15,
    DROP_FAIL: 20
};

let state = {
    // Infrastructure
    video: null,
    canvas: null,
    ctx: null,
    landmarker: null,
    
    // Flags
    isModelLoaded: false,
    isVideoReady: false,
    isTestRunning: false,
    
    // Logic Data
    prevWrist: null,
    phase: 'IDLE',
    repCount: 0,
    currentRepPeak: 0,
    repHistory: [],
    baseline: 0,
    
    // Visual Cache (To redraw lines when paused)
    lastLandmarks: null
};

async function initializeApp() {
    try {
        state.video = document.getElementById('video');
        state.canvas = document.getElementById('canvas');
        state.ctx = state.canvas.getContext('2d');
        
        console.log("🚀 Starting App...");
        
        const visionGen = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        state.landmarker = await PoseLandmarker.createFromOptions(visionGen, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`,
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numPoses: 1
        });

        state.isModelLoaded = true;
        document.getElementById('loading-overlay').classList.add('hidden');
        document.getElementById('status-pill').textContent = "Ready";

        // Wiring
        document.getElementById('btn-camera').onclick = startCamera;
        document.getElementById('file-input').onchange = handleUpload;
        
        state.video.addEventListener('loadeddata', onVideoReady);
        state.video.addEventListener('seeked', forceSingleScan); // Force scan on reset/scrub
        
        setupControls();
        
        // START PERMANENT UI LOOP
        requestAnimationFrame(uiLoop);

    } catch (e) { alert(e.message); }
}

function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    hardResetState();
    state.isVideoReady = false; 
    document.getElementById('status-pill').textContent = "Loading...";
    state.video.src = URL.createObjectURL(file);
    state.video.load(); 
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
        hardResetState();
        state.video.srcObject = stream;
        state.video.src = "";
    } catch (e) { alert(e.message); }
}

function onVideoReady() {
    state.isVideoReady = true;
    state.canvas.width = state.video.videoWidth;
    state.canvas.height = state.video.videoHeight;
    forceSingleScan(); // Scan frame 0 immediately
    document.getElementById('btn-start-test').disabled = false;
    document.getElementById('status-pill').textContent = "Video Loaded";
}

function forceSingleScan() {
    if(!state.isModelLoaded || !state.isVideoReady) return;
    // Process current frame (even if paused)
    const timeMs = state.video.currentTime * 1000;
    const result = state.landmarker.detectForVideo(state.video, timeMs);
    if(result.landmarks && result.landmarks.length > 0) {
        state.lastLandmarks = result.landmarks[0]; // Cache for UI Loop
    }
}

// --- THE PERMANENT LOOP ---
// Runs 60fps regardless of video state to keep lines/dots visible
function uiLoop() {
    // 1. If playing, run detection (AI)
    if (!state.video.paused && state.isModelLoaded && state.isVideoReady) {
        const result = state.landmarker.detectForVideo(state.video, performance.now());
        if(result.landmarks && result.landmarks.length > 0) {
            state.lastLandmarks = result.landmarks[0];
            
            // Run Physics ONLY during playback
            if(state.isTestRunning) {
                runBiomechanics(state.lastLandmarks, state.video.currentTime * 1000);
            }
        }
    }

    // 2. Always Draw (Visuals)
    drawEverything();
    
    requestAnimationFrame(uiLoop);
}

function drawEverything() {
    // Clear
    state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
    
    if (state.lastLandmarks) {
        const pose = state.lastLandmarks;
        const wrist = pose[CONFIG.WRIST_ID];
        const shoulder = pose[CONFIG.SHOULDER_ID];
        const hip = pose[CONFIG.HIP_ID];

        if (wrist && shoulder && hip) {
            drawZones(shoulder, hip);
            drawDot(wrist);
        }
    }
}

function runBiomechanics(pose, time) {
    const wrist = pose[CONFIG.WRIST_ID];
    const shoulder = pose[CONFIG.SHOULDER_ID];
    const hip = pose[CONFIG.HIP_ID];
    
    if (!state.prevWrist) {
        state.prevWrist = { x: wrist.x, y: wrist.y, time: time };
        return;
    }

    const dt = (time - state.prevWrist.time) / 1000;
    if (dt <= 0.001) return;

    // Calc Velocity
    const dx = (wrist.x - state.prevWrist.x) * state.canvas.width;
    const dy = (wrist.y - state.prevWrist.y) * state.canvas.height;
    const distPx = Math.sqrt(dx*dx + dy*dy);
    const bodySegmentPx = Math.abs(shoulder.y - hip.y) * state.canvas.height;
    const pixelsPerMeter = bodySegmentPx / 0.5; 
    const velocity = (distPx / pixelsPerMeter) / dt;

    document.getElementById('val-velocity').textContent = velocity.toFixed(2);
    
    // State Machine
    const isBelowHip = wrist.y > hip.y;
    const isAboveShoulder = wrist.y < shoulder.y;
    
    if (state.phase === 'IDLE' || state.phase === 'LOCKOUT') {
        if (isBelowHip) {
            state.phase = 'BOTTOM';
            document.getElementById('val-velocity').style.color = "#fbbf24";
        }
    } else if (state.phase === 'BOTTOM') {
        if (wrist.y < hip.y) {
            state.phase = 'CONCENTRIC';
            state.currentRepPeak = 0;
            document.getElementById('val-velocity').style.color = "#3b82f6";
        }
    } else if (state.phase === 'CONCENTRIC') {
        if (velocity > state.currentRepPeak) state.currentRepPeak = velocity;
        
        if (isAboveShoulder && velocity < 0.5) finishRep();
        else if (isBelowHip) state.phase = 'BOTTOM';
    }

    state.prevWrist = { x: wrist.x, y: wrist.y, time: time };
}

function finishRep() {
    state.phase = 'LOCKOUT';
    state.repCount++;
    state.repHistory.push(state.currentRepPeak);
    
    document.getElementById('val-reps').textContent = state.repCount;
    document.getElementById('val-peak').textContent = state.currentRepPeak.toFixed(2);
    document.getElementById('val-velocity').style.color = "#10b981";

    if (state.repCount <= 3) {
        const sum = state.repHistory.reduce((a, b) => a + b, 0);
        state.baseline = sum / state.repHistory.length;
        document.getElementById('val-drop').textContent = "CALC...";
    } else {
        const drop = (state.baseline - state.currentRepPeak) / state.baseline;
        const dropPct = (drop * 100).toFixed(1);
        const dropEl = document.getElementById('val-drop');
        dropEl.textContent = `-${dropPct}%`;
        
        if (drop * 100 >= CONFIG.DROP_FAIL) dropEl.style.color = "#ef4444";
        else if (drop * 100 >= CONFIG.DROP_WARN) dropEl.style.color = "#fbbf24";
        else dropEl.style.color = "#10b981";
    }
}

function drawZones(shoulder, hip) {
    const ctx = state.ctx;
    const w = state.canvas.width;
    const h = state.canvas.height;
    
    const shoulderY = shoulder.y * h;
    ctx.strokeStyle = "rgba(16, 185, 129, 0.5)"; 
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, shoulderY);
    ctx.lineTo(w, shoulderY);
    ctx.stroke();
    
    const hipY = hip.y * h;
    ctx.strokeStyle = "rgba(59, 130, 246, 0.5)"; 
    ctx.beginPath();
    ctx.moveTo(0, hipY);
    ctx.lineTo(w, hipY);
    ctx.stroke();
}

function drawDot(wrist) {
    const x = wrist.x * state.canvas.width;
    const y = wrist.y * state.canvas.height;
    state.ctx.fillStyle = state.phase === 'CONCENTRIC' ? "#3b82f6" : "#ef4444";
    state.ctx.beginPath();
    state.ctx.arc(x, y, 10, 0, 2*Math.PI);
    state.ctx.fill();
}

function hardResetState() {
    state.isTestRunning = false;
    state.prevWrist = null;
    state.repCount = 0;
    state.phase = 'IDLE';
    state.repHistory = [];
    state.baseline = 0;
    // Don't clear lastLandmarks here, so we still see the pose while waiting
    
    document.getElementById('val-velocity').textContent = "0.00";
    document.getElementById('val-peak').textContent = "0.00";
    document.getElementById('val-reps').textContent = "0";
    document.getElementById('val-drop').textContent = "--";
    
    document.getElementById('btn-start-test').textContent = "▶ Start Test";
    document.getElementById('btn-start-test').disabled = false;
    document.getElementById('btn-reset').disabled = true;
}

function setupControls() {
    const startBtn = document.getElementById('btn-start-test');
    const resetBtn = document.getElementById('btn-reset');

    startBtn.onclick = () => {
        state.isTestRunning = true;
        startBtn.textContent = "Test Running...";
        startBtn.disabled = true;
        resetBtn.disabled = false;
        
        state.phase = 'IDLE'; 
        state.repCount = 0;
        state.repHistory = [];
        
        state.video.play();
    };

    resetBtn.onclick = () => {
        state.video.pause();
        hardResetState();
        state.video.currentTime = 0; // Triggers 'seeked' -> forceSingleScan
    };
}

initializeApp();
