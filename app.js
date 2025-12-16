import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const CONFIG = {
    // LANDMARKS (Right Side Defaults)
    WRIST_ID: 16,
    SHOULDER_ID: 12,
    HIP_ID: 24,
    
    // PHYSICS
    SMOOTHING: 3,
    MIN_ROM: 0.3, // Minimum vertical Range of Motion (meters) to count as a rep
    
    // ALERTS
    DROP_WARN: 15, // 15% drop
    DROP_FAIL: 20  // 20% drop
};

let state = {
    // Infrastructure
    video: null,
    canvas: null,
    ctx: null,
    landmarker: null,
    isModelLoaded: false,
    isVideoReady: false,
    isTestRunning: false,
    
    // Tracking Data
    prevWrist: null,
    velocityBuffer: [],
    
    // RECOVERY / COACHING STATE
    phase: 'IDLE', // Phases: IDLE -> BOTTOM -> CONCENTRIC -> LOCKOUT
    repCount: 0,
    currentRepPeak: 0,
    repHistory: [],
    baseline: 0
};

async function initializeApp() {
    try {
        state.video = document.getElementById('video');
        state.canvas = document.getElementById('canvas');
        state.ctx = state.canvas.getContext('2d');
        
        console.log("🚀 Starting Biomechanics Engine...");
        
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

        document.getElementById('btn-camera').onclick = startCamera;
        document.getElementById('file-input').onchange = handleUpload;
        
        state.video.addEventListener('loadeddata', onVideoReady);
        state.video.addEventListener('seeked', () => {
             if(state.isVideoReady) processFrame(state.video.currentTime * 1000);
        });
        
        setupControls();

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
    processFrame(0); 
    document.getElementById('btn-start-test').disabled = false;
    document.getElementById('status-pill').textContent = "Video Loaded";
}

function renderLoop(now, metadata) {
    if (!state.video.paused) {
        processFrame(metadata.mediaTime * 1000);
        state.video.requestVideoFrameCallback(renderLoop);
    }
}

// --- NEW PROCESSING LOGIC ---
function processFrame(timeMs) {
    if (!state.isModelLoaded || !state.isVideoReady) return;

    const result = state.landmarker.detectForVideo(state.video, timeMs);
    state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
    
    if (result.landmarks && result.landmarks.length > 0) {
        const pose = result.landmarks[0];
        
        // Extract Key Points
        const wrist = pose[CONFIG.WRIST_ID];
        const shoulder = pose[CONFIG.SHOULDER_ID];
        const hip = pose[CONFIG.HIP_ID];

        if (wrist && shoulder && hip) {
            // Visualize Zones
            drawZones(shoulder, hip);
            drawDot(wrist);
            
            if (state.isTestRunning) {
                runBiomechanicsEngine(wrist, shoulder, hip, timeMs);
            }
        }
    }
}

function runBiomechanicsEngine(wrist, shoulder, hip, time) {
    // 1. Calculate Velocity (Standard VBT Math)
    if (!state.prevWrist) {
        state.prevWrist = { x: wrist.x, y: wrist.y, time: time };
        return;
    }

    const dt = (time - state.prevWrist.time) / 1000;
    if (dt <= 0.0001 || dt > 1.0) return; // Guard

    // Convert normalized coords to pixels
    const dx = (wrist.x - state.prevWrist.x) * state.canvas.width;
    const dy = (wrist.y - state.prevWrist.y) * state.canvas.height;
    const distPx = Math.sqrt(dx*dx + dy*dy);
    
    // Calibration (Auto-scale: Assume distance from Shoulder to Hip is 0.5m)
    // This is better than fixed width calibration as it scales with the subject
    const bodySegmentPx = Math.abs(shoulder.y - hip.y) * state.canvas.height;
    const pixelsPerMeter = bodySegmentPx / 0.5; // Approx 0.5m torso length
    
    const velocity = (distPx / pixelsPerMeter) / dt;

    // Update Live View
    document.getElementById('val-velocity').textContent = velocity.toFixed(2);
    
    // 2. STATE MACHINE (The "Snatch" Logic)
    // Y-Coordinate: 0 is TOP, 1 is BOTTOM.
    // So "Higher than shoulder" means wrist.y < shoulder.y
    
    const isBelowHip = wrist.y > hip.y;
    const isAboveShoulder = wrist.y < shoulder.y;
    
    // STATE 1: WAITING IN THE HOLE (Backswing/Start)
    if (state.phase === 'IDLE' || state.phase === 'LOCKOUT') {
        if (isBelowHip) {
            state.phase = 'BOTTOM';
            document.getElementById('val-velocity').style.color = "#fbbf24"; // Yellow (Ready)
        }
    }
    
    // STATE 2: EXPLOSION (Concentric)
    else if (state.phase === 'BOTTOM') {
        // If we move UP past the hip, the rep has started
        if (wrist.y < hip.y) {
            state.phase = 'CONCENTRIC';
            state.currentRepPeak = 0; // Reset peak for this new rep
            document.getElementById('val-velocity').style.color = "#3b82f6"; // Blue (Go!)
        }
    }
    
    // STATE 3: TRACKING THE PEAK
    else if (state.phase === 'CONCENTRIC') {
        // Track Max Velocity during upward phase
        if (velocity > state.currentRepPeak) {
            state.currentRepPeak = velocity;
        }
        
        // STATE 4: LOCKOUT (Finish)
        // Must be above shoulder AND velocity must drop (pause overhead)
        if (isAboveShoulder && velocity < 0.5) {
            finishRep();
        }
        // Fail-safe: If they drop the bell back below hip without locking out
        else if (isBelowHip) {
            state.phase = 'BOTTOM'; // Reset without counting rep
        }
    }

    state.prevWrist = { x: wrist.x, y: wrist.y, time: time };
}

function finishRep() {
    state.phase = 'LOCKOUT';
    state.repCount++;
    state.repHistory.push(state.currentRepPeak);
    
    // Update UI
    document.getElementById('val-reps').textContent = state.repCount;
    document.getElementById('val-peak').textContent = state.currentRepPeak.toFixed(2);
    document.getElementById('val-velocity').style.color = "#10b981"; // Green (Good Rep)

    // Calculate Drop-off
    if (state.repCount <= 3) {
        // Build Baseline
        const sum = state.repHistory.reduce((a, b) => a + b, 0);
        state.baseline = sum / state.repHistory.length;
        document.getElementById('val-drop').textContent = "CALC...";
    } else {
        // Compare to Baseline
        const drop = (state.baseline - state.currentRepPeak) / state.baseline;
        const dropPct = (drop * 100).toFixed(1);
        const dropEl = document.getElementById('val-drop');
        dropEl.textContent = `-${dropPct}%`;
        
        if (drop * 100 >= CONFIG.DROP_FAIL) dropEl.style.color = "#ef4444";
        else if (drop * 100 >= CONFIG.DROP_WARN) dropEl.style.color = "#fbbf24";
        else dropEl.style.color = "#10b981";
    }
}

// --- VISUALS ---
function drawZones(shoulder, hip) {
    const ctx = state.ctx;
    const w = state.canvas.width;
    const h = state.canvas.height;
    
    // Draw "Lockout Line" (Shoulder Height)
    const shoulderY = shoulder.y * h;
    ctx.strokeStyle = "rgba(16, 185, 129, 0.3)"; // Green Line
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, shoulderY);
    ctx.lineTo(w, shoulderY);
    ctx.stroke();
    
    // Draw "Start Line" (Hip Height)
    const hipY = hip.y * h;
    ctx.strokeStyle = "rgba(59, 130, 246, 0.3)"; // Blue Line
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
    
    state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
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
        
        state.phase = 'IDLE'; // Reset phase logic
        state.repCount = 0;
        state.repHistory = [];
        
        state.video.play();
        state.video.requestVideoFrameCallback(renderLoop);
    };

    resetBtn.onclick = () => {
        state.video.pause();
        hardResetState();
        state.video.currentTime = 0;
    };
}

initializeApp();
