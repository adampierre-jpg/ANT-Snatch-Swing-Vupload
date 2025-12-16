import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const CONFIG = {
    WRIST_INDEX: 16, 
    CONFIDENCE: 0.3,
    TRACKING_CONFIDENCE: 0.3,
    SMOOTHING: 0 
};

let state = {
    video: null,
    canvas: null,
    ctx: null,
    landmarker: null,
    
    // Flags
    isModelLoaded: false,
    isVideoReady: false,
    isTestRunning: false,
    
    // Data
    prevWrist: null,
    maxVelocity: 0
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
            numPoses: 1,
            minPoseDetectionConfidence: CONFIG.CONFIDENCE,
            minTrackingConfidence: CONFIG.TRACKING_CONFIDENCE
        });

        console.log("✅ AI Model Loaded");
        state.isModelLoaded = true;
        document.getElementById('loading-overlay').classList.add('hidden');
        document.getElementById('status-pill').textContent = "Select Video Source";

        // Inputs
        document.getElementById('btn-camera').onclick = startCamera;
        document.getElementById('file-input').onchange = handleUpload;
        
        // --- EVENT LISTENERS ---
        // 'loadeddata': Fires when video is ready to play
        state.video.addEventListener('loadeddata', onVideoReady);
        // 'seeked': Fires when user scrubs or we rewind
        state.video.addEventListener('seeked', () => {
            // Only process if video is ready (avoids error on empty source)
            if (state.isVideoReady) {
                processSingleFrame(state.video.currentTime * 1000);
            }
        });
        
        setupControls();

    } catch (e) {
        console.error(e);
        alert("Startup Error: " + e.message);
    }
}

/**
 * SOURCE HANDLING
 */
function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    console.log("📂 File Selected");
    
    // 1. NUKE OLD STATE
    hardResetState();
    
    // 2. Set Loading UI
    state.isVideoReady = false; 
    document.getElementById('status-pill').textContent = "Loading Video...";
    document.getElementById('btn-start-test').disabled = true;
    
    // 3. Load New Source
    const url = URL.createObjectURL(file);
    state.video.src = url;
    state.video.load(); 
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
        hardResetState();
        state.video.srcObject = stream;
        state.video.src = "";
    } catch (e) {
        alert("Camera Error: " + e.message);
    }
}

function onVideoReady() {
    console.log("✅ Video Loaded");
    state.isVideoReady = true;
    
    // Resize canvas
    state.canvas.width = state.video.videoWidth;
    state.canvas.height = state.video.videoHeight;
    
    // Force immediate scan of frame 0
    processSingleFrame(0); 
    
    document.getElementById('btn-start-test').disabled = false;
    document.getElementById('status-pill').textContent = "Ready to Test";
}

/**
 * CORE LOGIC
 */
function processSingleFrame(timeMs) {
    if (!state.isModelLoaded || !state.isVideoReady) return;

    // Detect
    const result = state.landmarker.detectForVideo(state.video, timeMs);
    
    // Clear Canvas
    state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
    
    if (result.landmarks && result.landmarks.length > 0) {
        const wrist = result.landmarks[0][CONFIG.WRIST_INDEX];
        if (wrist) {
            drawDot(wrist);
            
            // Update UI Status if not running
            const status = document.getElementById('status-pill');
            if (!state.isTestRunning) {
                status.textContent = "Wrist Found ✓";
                status.style.color = "#10b981";
            }

            // Physics (Only if Test Running)
            if (state.isTestRunning) {
                calculatePhysics(wrist, timeMs);
            }
        }
    } else {
        // Show scanning if idle
        if (!state.isTestRunning) {
            const status = document.getElementById('status-pill');
            status.textContent = "Scanning...";
            status.style.color = "#fbbf24";
        }
    }
}

function renderLoop(now, metadata) {
    if (!state.video.paused) {
        const videoTimeMs = metadata.mediaTime * 1000;
        processSingleFrame(videoTimeMs);
        state.video.requestVideoFrameCallback(renderLoop);
    }
}

function calculatePhysics(wrist, time) {
    if (!state.prevWrist) {
        state.prevWrist = { x: wrist.x, y: wrist.y, time: time };
        return;
    }

    const dt = (time - state.prevWrist.time) / 1000;
    
    // Guard: Skip duplicate frames or huge jumps (rewinds)
    if (dt <= 0.0001 || dt > 1.0) return;

    const dx = (wrist.x - state.prevWrist.x) * state.canvas.width;
    const dy = (wrist.y - state.prevWrist.y) * state.canvas.height;
    const distPx = Math.sqrt(dx*dx + dy*dy);
    
    // Calibration: 2m width
    const metersPerPixel = 2.0 / state.canvas.width; 
    const velocity = (distPx * metersPerPixel) / dt;

    if (velocity > state.maxVelocity && velocity < 100) {
        state.maxVelocity = velocity;
        document.getElementById('val-velocity').textContent = state.maxVelocity.toFixed(2);
        // document.getElementById('val-velocity').style.color = "red"; // Optional color pop
    }
    
    state.prevWrist = { x: wrist.x, y: wrist.y, time: time };
}

function drawDot(wrist) {
    const x = wrist.x * state.canvas.width;
    const y = wrist.y * state.canvas.height;
    state.ctx.fillStyle = "#ef4444";
    state.ctx.beginPath();
    state.ctx.arc(x, y, 10, 0, 2*Math.PI);
    state.ctx.fill();
}

/**
 * CONTROLS & RESET LOGIC
 */
function hardResetState() {
    // 1. Reset Flags
    state.isTestRunning = false;
    state.prevWrist = null;
    state.maxVelocity = 0;
    
    // 2. Clear Visuals
    state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
    
    // 3. Reset UI Elements
    document.getElementById('val-velocity').textContent = "0.00";
    document.getElementById('val-velocity').style.color = "white"; // Reset color
    
    document.getElementById('status-pill').textContent = "Ready";
    document.getElementById('status-pill').style.color = "white"; // Reset color
    
    const startBtn = document.getElementById('btn-start-test');
    const resetBtn = document.getElementById('btn-reset');
    
    startBtn.textContent = "▶ Start Test";
    startBtn.disabled = false;
    resetBtn.disabled = true;
}

function setupControls() {
    const startBtn = document.getElementById('btn-start-test');
    const resetBtn = document.getElementById('btn-reset');

    startBtn.onclick = () => {
        // Lock UI
        state.isTestRunning = true;
        startBtn.textContent = "Test Running...";
        startBtn.disabled = true;
        resetBtn.disabled = false;
        
        // Reset Physics for this specific run
        state.prevWrist = null;
        state.maxVelocity = 0;
        document.getElementById('val-velocity').textContent = "0.00";
        
        // Play
        state.video.play();
        state.video.requestVideoFrameCallback(renderLoop);
    };

    resetBtn.onclick = () => {
        state.video.pause();
        
        // NUKE STATE
        hardResetState();
        
        // Rewind (Triggers 'seeked' -> processSingleFrame(0))
        state.video.currentTime = 0;
    };
}

initializeApp();
