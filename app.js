import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const CONFIG = {
    WRIST_INDEX: 16, // Right wrist (use 15 for left)
    // LOWERED CONFIDENCE: Allows tracking even when arm is blurry
    CONFIDENCE: 0.3, 
    TRACKING_CONFIDENCE: 0.3,
    // Removed smoothing for raw peak detection
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
    maxVelocity: 0 // New metric to capture the peak
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

        // FIX 1: Use the FULL model (better accuracy for fast motion)
        // switched from 'pose_landmarker_lite' to 'pose_landmarker_full'
        state.landmarker = await PoseLandmarker.createFromOptions(visionGen, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`,
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numPoses: 1,
            // FIX 2: Lower thresholds to accept motion-blurred limbs
            minPoseDetectionConfidence: CONFIG.CONFIDENCE,
            minTrackingConfidence: CONFIG.TRACKING_CONFIDENCE
        });

        console.log("✅ AI Model Loaded (Full Version)");
        state.isModelLoaded = true;
        document.getElementById('loading-overlay').classList.add('hidden');
        document.getElementById('status-pill').textContent = "Select Video Source";

        // Inputs
        document.getElementById('btn-camera').onclick = startCamera;
        document.getElementById('file-input').onchange = handleUpload;
        
        state.video.addEventListener('loadeddata', onVideoReady);
        
        // Button Logic
        const startBtn = document.getElementById('btn-start-test');
        const resetBtn = document.getElementById('btn-reset');

        startBtn.onclick = () => {
            state.isTestRunning = true;
            state.prevWrist = null;
            state.maxVelocity = 0; // Reset peak
            startBtn.disabled = true;
            resetBtn.disabled = false;
            
            if (state.video.paused) state.video.play();
        };

        resetBtn.onclick = () => {
            state.isTestRunning = false;
            state.prevWrist = null;
            state.ctx.clearRect(0,0,state.canvas.width, state.canvas.height);
            startBtn.disabled = false;
            resetBtn.disabled = true;
            document.getElementById('val-velocity').textContent = "0.00";
            state.video.pause();
            state.video.currentTime = 0;
        };
        
        // FIX 3: Start the optimized render loop
        if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
            state.video.requestVideoFrameCallback(renderLoop);
        } else {
            alert("Your browser does not support frame-by-frame sync. Use Chrome.");
            requestAnimationFrame(legacyRenderLoop);
        }

    } catch (e) {
        console.error(e);
        alert("Startup Error: " + e.message);
    }
}

// ... (handleUpload and startCamera remain the same) ...
function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
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
    state.isVideoReady = true;
    state.canvas.width = state.video.videoWidth;
    state.canvas.height = state.video.videoHeight;
}

// FIX 3: The Synchronized Loop
function renderLoop(now, metadata) {
    // metadata.mediaTime provides the EXACT timestamp of the frame
    if (state.isModelLoaded && state.isVideoReady && !state.video.paused) {
        
        // Use metadata.mediaTime instead of video.currentTime for sub-millisecond precision
        const startTimeMs = metadata.mediaTime * 1000;

        const result = state.landmarker.detectForVideo(state.video, startTimeMs);
        state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
        
        if (result.landmarks && result.landmarks.length > 0) {
            const wrist = result.landmarks[0][CONFIG.WRIST_INDEX];
            if (wrist) {
                drawDot(wrist);
                if (state.isTestRunning) {
                    calculatePhysics(wrist, startTimeMs);
                }
            }
        }
    }
    // Re-register the callback for the next video frame
    state.video.requestVideoFrameCallback(renderLoop);
}

function calculatePhysics(wrist, time) {
    // 1. Initialization
    if (!state.prevWrist) {
        state.prevWrist = { x: wrist.x, y: wrist.y, time: time };
        return;
    }

    // 2. Calculate time delta in seconds
    const dt = (time - state.prevWrist.time) / 1000;
    
    // Safety check: if dt is 0 (duplicate frame) or massive (seek), skip
    if (dt <= 0.0001 || dt > 1.0) return; 

    // 3. Pixel Distance
    const dx = (wrist.x - state.prevWrist.x) * state.canvas.width;
    const dy = (wrist.y - state.prevWrist.y) * state.canvas.height;
    const distPx = Math.sqrt(dx*dx + dy*dy);
    
    // 4. Calibration (Assume width = 2 meters for now)
    // NOTE: This is highly sensitive to camera distance!
    const metersPerPixel = 2.0 / state.canvas.width; 
    const velocity = (distPx * metersPerPixel) / dt;

    // 5. Peak Detection (No Smoothing for Ballistic)
    if (velocity > state.maxVelocity && velocity < 100) { // < 100 sanity check
        state.maxVelocity = velocity;
        document.getElementById('val-velocity').textContent = state.maxVelocity.toFixed(2);
        
        // Visual Feedback for Peak
        document.getElementById('val-velocity').style.color = "red";
        setTimeout(() => document.getElementById('val-velocity').style.color = "black", 200);
    }
    
    state.prevWrist = { x: wrist.x, y: wrist.y, time: time };
}

function drawDot(wrist) {
    const x = wrist.x * state.canvas.width;
    const y = wrist.y * state.canvas.height;
    state.ctx.fillStyle = "#ef4444";
    state.ctx.beginPath();
    state.ctx.arc(x, y, 8, 0, 2*Math.PI);
    state.ctx.fill();
}

initializeApp();
