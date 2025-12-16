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
        
        // --- CRITICAL FIX: Event Listeners for Static Detection ---
        // These run even when video is PAUSED
        state.video.addEventListener('loadeddata', onVideoReady);
        state.video.addEventListener('seeked', () => processSingleFrame(state.video.currentTime * 1000));
        
        setupControls();

    } catch (e) {
        console.error(e);
        alert("Startup Error: " + e.message);
    }
}

function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    console.log("📂 File Selected");
    state.isVideoReady = false; 
    document.getElementById('status-pill').textContent = "Loading...";
    
    const url = URL.createObjectURL(file);
    state.video.src = url;
    state.video.load(); 
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
        state.video.srcObject = stream;
        state.video.src = "";
    } catch (e) {
        alert("Camera Error: " + e.message);
    }
}

function onVideoReady() {
    console.log("✅ Video Loaded");
    state.isVideoReady = true;
    
    // Resize canvas to match video
    state.canvas.width = state.video.videoWidth;
    state.canvas.height = state.video.videoHeight;
    
    // CRITICAL FIX: Force an immediate scan of the first frame
    // We don't wait for "play" to happen.
    processSingleFrame(0); 
    
    document.getElementById('btn-start-test').disabled = false;
}

// --- NEW FUNCTION: Processes one frame (used by Loop AND Events) ---
function processSingleFrame(timeMs) {
    if (!state.isModelLoaded || !state.isVideoReady) return;

    // Detect
    const result = state.landmarker.detectForVideo(state.video, timeMs);
    
    // Clear & Draw
    state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
    
    if (result.landmarks && result.landmarks.length > 0) {
        const wrist = result.landmarks[0][CONFIG.WRIST_INDEX];
        if (wrist) {
            drawDot(wrist);
            
            // Update UI Status
            const status = document.getElementById('status-pill');
            if(status.textContent !== "Test Running...") {
                status.textContent = "Wrist Found ✓";
                status.style.color = "#10b981";
            }

            // Only calculate physics if the TEST is actually running
            if (state.isTestRunning) {
                calculatePhysics(wrist, timeMs);
            }
        }
    } else {
        // Only show "Scanning" if we aren't mid-test (avoids flickering)
        if (!state.isTestRunning) {
            document.getElementById('status-pill').textContent = "Scanning...";
            document.getElementById('status-pill').style.color = "#fbbf24";
        }
    }
}

// --- THE PHYSICS LOOP (Only runs during playback) ---
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
    if (dt <= 0.0001 || dt > 1.0) return; // Skip bad frames

    const dx = (wrist.x - state.prevWrist.x) * state.canvas.width;
    const dy = (wrist.y - state.prevWrist.y) * state.canvas.height;
    const distPx = Math.sqrt(dx*dx + dy*dy);
    
    // Calibration: 2m width
    const metersPerPixel = 2.0 / state.canvas.width; 
    const velocity = (distPx * metersPerPixel) / dt;

    if (velocity > state.maxVelocity && velocity < 100) {
        state.maxVelocity = velocity;
        document.getElementById('val-velocity').textContent = state.maxVelocity.toFixed(2);
        document.getElementById('val-velocity').style.color = "red";
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

function setupControls() {
    const startBtn = document.getElementById('btn-start-test');
    const resetBtn = document.getElementById('btn-reset');

    startBtn.onclick = () => {
        state.isTestRunning = true;
        state.prevWrist = null;
        state.maxVelocity = 0;
        
        startBtn.textContent = "Test Running...";
        startBtn.disabled = true;
        resetBtn.disabled = false;
        
        state.video.play();
        state.video.requestVideoFrameCallback(renderLoop);
    };

    resetBtn.onclick = () => {
        state.isTestRunning = false;
        state.prevWrist = null;
        startBtn.textContent = "▶ Start Test";
        startBtn.disabled = false;
        resetBtn.disabled = true;
        
        state.video.pause();
        state.video.currentTime = 0;
        // Force one update on reset to show the dot again
        setTimeout(() => processSingleFrame(0), 100);
    };
}

initializeApp();
