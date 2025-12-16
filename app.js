import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

/**
 * ============================================================================
 * 1. CONFIGURATION & STATE
 * ============================================================================
 */
const CONFIG = {
    // SYSTEM
    VIDEO_WIDTH: 640,
    VIDEO_HEIGHT: 480,
    LANDMARK_WRIST: 16, // Right wrist index
    
    // PHYSICS
    SHOULDER_WIDTH_M: 0.45, // Calibration reference (meters)
    SMOOTHING_WINDOW: 3     // Rolling average frames
};

let state = {
    // DOM Elements
    video: null,
    canvas: null,
    ctx: null,
    
    // AI & System
    landmarker: null,
    isReady: false,
    
    // Runtime Physics
    lastFrameTime: 0,     // Timestamp of last processed frame
    prevWrist: null,      // {x, y, time} from previous frame
    pixelScale: 0.002,    // Meters per Pixel (will auto-calibrate)
    
    // Metrics
    currentVelocity: 0,
    velocityHistory: [],  // For smoothing
};

/**
 * ============================================================================
 * 2. INITIALIZATION
 * ============================================================================
 */
async function initializeApp() {
    try {
        // 1. Setup DOM Elements
        state.video = document.getElementById('video');
        state.canvas = document.getElementById('canvas');
        state.ctx = state.canvas.getContext('2d');
        
        // 2. Initialize MediaPipe
        console.log("⏳ Loading MediaPipe...");
        const visionGen = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        
        state.landmarker = await PoseLandmarker.createFromOptions(visionGen, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numPoses: 1
        });

        // 3. UI Ready State
        console.log("✅ MediaPipe Ready");
        document.getElementById('loading-overlay').classList.add('hidden');
        document.getElementById('status-pill').textContent = "System Ready";

        // 4. Attach Listeners
        document.getElementById('btn-camera').addEventListener('click', startCameraMode);
        document.getElementById('video-upload').addEventListener('change', startUploadMode);
        
        // 5. Setup Video Controls
        setupVideoControls();

    } catch (error) {
        console.error(error);
        alert("Startup Error: " + error.message);
    }
}

/**
 * ============================================================================
 * 3. VIDEO CONTROLS LOGIC
 * ============================================================================
 */
function setupVideoControls() {
    const playBtn = document.getElementById('btn-play-pause');
    const seekBar = document.getElementById('seek-bar');
    const timeDisplay = document.getElementById('time-display');
    const video = state.video;

    // Play/Pause
    playBtn.addEventListener('click', () => {
        if (video.paused) {
            video.play();
            playBtn.textContent = "⏸";
        } else {
            video.pause();
            playBtn.textContent = "▶";
        }
    });

    // Seek Bar Interaction
    seekBar.addEventListener('input', () => {
        video.currentTime = seekBar.value;
    });

    // Update UI as video plays
    video.addEventListener('timeupdate', () => {
        if (!isNaN(video.duration)) {
            seekBar.max = video.duration;
            seekBar.value = video.currentTime;
            
            // Format 0:00
            const mins = Math.floor(video.currentTime / 60);
            const secs = Math.floor(video.currentTime % 60).toString().padStart(2, '0');
            timeDisplay.textContent = `${mins}:${secs}`;
        }
    });
}

/**
 * ============================================================================
 * 4. INPUT MODES (CAMERA vs UPLOAD)
 * ============================================================================
 */
async function startCameraMode() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: CONFIG.VIDEO_WIDTH, height: CONFIG.VIDEO_HEIGHT, facingMode: 'user' } 
        });
        state.video.srcObject = stream;
        state.video.src = "";
        state.video.loop = false;
        
        // Camera: Disable custom seek bar (can't seek live video)
        document.getElementById('seek-bar').disabled = true;
        
        await playVideo();
    } catch (e) {
        alert("Camera Error: " + e.message);
    }
}

function startUploadMode(event) {
    const file = event.target.files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    state.video.srcObject = null;
    state.video.src = url;
    state.video.loop = false;
    
    // Upload: Enable seek bar
    document.getElementById('seek-bar').disabled = false;
    
    playVideo();
}

async function playVideo() {
    return new Promise((resolve) => {
        state.video.onloadeddata = () => {
            state.canvas.width = state.video.videoWidth;
            state.canvas.height = state.video.videoHeight;
            state.video.play();
            document.getElementById('btn-play-pause').textContent = "⏸";
            
            if (!state.isReady) {
                state.isReady = true;
                window.requestAnimationFrame(renderLoop);
            }
            resolve();
        };
    });
}

/**
 * ============================================================================
 * 5. MAIN LOOP & PHYSICS
 * ============================================================================
 */
function renderLoop() {
    if (!state.isReady) return;

    // Check if frame has advanced
    if (!state.video.paused && !state.video.ended && state.video.currentTime !== state.lastFrameTime) {
        
        let startTimeMs = performance.now();
        if (state.video.duration) {
             startTimeMs = state.video.currentTime * 1000;
        }

        // A. DETECT POSE
        const result = state.landmarker.detectForVideo(state.video, startTimeMs);
        
        // Clear Canvas
        state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
        
        if (result.landmarks && result.landmarks.length > 0) {
            const landmarks = result.landmarks[0];
            const wrist = landmarks[CONFIG.LANDMARK_WRIST];
            const leftShoulder = landmarks[11];
            const rightShoulder = landmarks[12];

            if (wrist) {
                // B. CALIBRATE (Auto-scale based on shoulder width)
                if (leftShoulder && rightShoulder) {
                    const shoulderDistPx = Math.hypot(
                        (leftShoulder.x - rightShoulder.x) * state.canvas.width,
                        (leftShoulder.y - rightShoulder.y) * state.canvas.height
                    );
                    if (shoulderDistPx > 0) {
                        state.pixelScale = CONFIG.SHOULDER_WIDTH_M / shoulderDistPx;
                    }
                }

                // C. CALCULATE VELOCITY
                calculateVelocity(wrist, startTimeMs);

                // D. DRAW VISUALS
                drawVisuals(wrist);
            }
        }
        state.lastFrameTime = state.video.currentTime;
    }
    
    window.requestAnimationFrame(renderLoop);
}

function calculateVelocity(wrist, timestamp) {
    const x = wrist.x * state.canvas.width;
    const y = wrist.y * state.canvas.height;
    
    if (state.prevWrist) {
        // Distance in Pixels
        const dx = x - state.prevWrist.x;
        const dy = y - state.prevWrist.y;
        const distPx = Math.sqrt(dx*dx + dy*dy);
        
        // Distance in Meters
        const distM = distPx * state.pixelScale;
        
        // Time Delta (seconds)
        const dt = (timestamp - state.prevWrist.time) / 1000;
        
        if (dt > 0.001) { // Avoid divide by zero
            const rawVelocity = distM / dt;
            
            // Smoothing (Rolling Average)
            state.velocityHistory.push(rawVelocity);
            if (state.velocityHistory.length > CONFIG.SMOOTHING_WINDOW) {
                state.velocityHistory.shift();
            }
            
            const smoothedVel = state.velocityHistory.reduce((a, b) => a + b) / state.velocityHistory.length;
            state.currentVelocity = smoothedVel;
            
            // UPDATE UI
            document.getElementById('val-velocity').textContent = smoothedVel.toFixed(2);
        }
    }
    
    // Store for next frame
    state.prevWrist = { x, y, time: timestamp };
}

function drawVisuals(wrist) {
    const ctx = state.ctx;
    const cx = wrist.x * state.canvas.width;
    const cy = wrist.y * state.canvas.height;

    // Draw Red Dot
    ctx.fillStyle = "#ef4444"; // Red-500
    ctx.beginPath();
    ctx.arc(cx, cy, 8, 0, 2 * Math.PI);
    ctx.fill();
    
    // Draw Velocity Text
    ctx.fillStyle = "white";
    ctx.font = "bold 16px sans-serif";
    ctx.fillText(`${state.currentVelocity.toFixed(2)} m/s`, cx + 15, cy);
}

// Start System
initializeApp();


