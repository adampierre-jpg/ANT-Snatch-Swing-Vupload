import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

/**
 * STATE MANAGEMENT
 */
const CONFIG = {
    WRIST_INDEX: 16,
    SHOULDER_WIDTH_M: 0.45,
    SMOOTHING_WINDOW: 3
};

let state = {
    // Infrastructure
    video: null,
    canvas: null,
    ctx: null,
    landmarker: null,
    isReady: false,
    
    // Physics & Logic
    pixelScale: 1,
    isTestRunning: false,
    lastFrameTime: 0,
    prevWrist: null,
    
    // Data
    velocityHistory: [],
    currentVelocity: 0,
    repCount: 0,
    repPeaks: []
};

/**
 * 1. INITIALIZATION
 */
async function initializeApp() {
    try {
        state.video = document.getElementById('video');
        state.canvas = document.getElementById('canvas');
        state.ctx = state.canvas.getContext('2d');

        console.log("⏳ Loading AI...");
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

        console.log("✅ AI Ready");
        document.getElementById('loading-overlay').classList.add('hidden');
        document.getElementById('status-pill').textContent = "Ready";

        // Wire Buttons
        document.getElementById('btn-camera').onclick = startCamera;
        document.getElementById('file-input').onchange = handleUpload;
        
        document.getElementById('btn-start-test').onclick = () => {
            state.isTestRunning = true;
            state.repCount = 0;
            state.velocityHistory = [];
            state.prevWrist = null;
            updateUI();
            
            const btn = document.getElementById('btn-start-test');
            btn.textContent = "Test Running...";
            btn.disabled = true;
            document.getElementById('btn-reset').disabled = false;
            
            if(state.video.paused) state.video.play();
        };

        document.getElementById('btn-reset').onclick = () => {
            state.isTestRunning = false;
            state.repCount = 0;
            state.currentVelocity = 0;
            updateUI();
            
            const btn = document.getElementById('btn-start-test');
            btn.textContent = "▶ Start Test";
            btn.disabled = false;
            document.getElementById('btn-reset').disabled = true;
            
            state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
        };

        setupVideoControls();

    } catch (e) {
        alert("Startup Error: " + e.message);
        console.error(e);
    }
}

/**
 * 2. SOURCE HANDLING
 */
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } } 
        });
        state.video.srcObject = stream;
        state.video.src = "";
        onVideoSourceChanged();
    } catch (e) {
        alert("Camera Error: " + e.message);
    }
}

function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    state.video.srcObject = null;
    state.video.src = URL.createObjectURL(file);
    onVideoSourceChanged();
}

function onVideoSourceChanged() {
    state.video.loop = false;
    state.video.onloadedmetadata = () => {
        // Sync Canvas Size to Video Source
        state.canvas.width = state.video.videoWidth;
        state.canvas.height = state.video.videoHeight;
        
        console.log(`Video Source Loaded: ${state.video.videoWidth}x${state.video.videoHeight}`);
        
        document.getElementById('btn-start-test').disabled = false;
        state.isReady = true;
        
        // Start Render Loop
        requestAnimationFrame(renderLoop);
    };
    state.video.play().catch(e => console.log("Auto-play blocked"));
}

/**
 * 3. MAIN PHYSICS LOOP
 */
function renderLoop() {
    if (!state.isReady) return;

    // Only process if video is playing and time has advanced
    if (!state.video.paused && !state.video.ended && state.video.currentTime !== state.lastFrameTime) {
        
        let startTimeMs = performance.now();
        if (state.video.duration) {
             startTimeMs = state.video.currentTime * 1000;
        }

        const result = state.landmarker.detectForVideo(state.video, startTimeMs);
        
        state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
        
        if (result.landmarks && result.landmarks.length > 0) {
            const landmarks = result.landmarks[0];
            const wrist = landmarks[CONFIG.WRIST_INDEX];
            
            if (wrist) {
                // Draw Tracking Dot
                drawDot(wrist);
                
                // Calculate Physics if Test is Running
                if (state.isTestRunning) {
                    processPhysics(wrist, startTimeMs);
                }
            }
        }
        state.lastFrameTime = state.video.currentTime;
    }
    
    requestAnimationFrame(renderLoop);
}

function processPhysics(wrist, time) {
    // Normalize coordinates to pixels
    const x = wrist.x * state.canvas.width;
    const y = wrist.y * state.canvas.height;
    
    if (state.prevWrist) {
        const dx = x - state.prevWrist.x;
        const dy = y - state.prevWrist.y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        
        // Time delta (seconds)
        const dt = (time - state.prevWrist.time) / 1000;
        
        // Simple Calibration (Assuming width = 1.5m visible area for now)
        // Ideally we calibrate on shoulders, but this is a robust fallback
        const pixelsPerMeter = state.canvas.width / 1.5; 
        
        if (dt > 0.01) {
            const vel = (dist / pixelsPerMeter) / dt;
            
            // Smoothing
            state.velocityHistory.push(vel);
            if (state.velocityHistory.length > CONFIG.SMOOTHING_WINDOW) state.velocityHistory.shift();
            
            state.currentVelocity = state.velocityHistory.reduce((a,b)=>a+b)/state.velocityHistory.length;
            
            // Simple Rep Detection (Velocity Zero-Crossing logic could go here)
            // For now, just show velocity
            updateUI();
        }
    }
    state.prevWrist = { x, y, time };
}

function drawDot(wrist) {
    const x = wrist.x * state.canvas.width;
    const y = wrist.y * state.canvas.height;
    state.ctx.fillStyle = "#ef4444";
    state.ctx.beginPath();
    state.ctx.arc(x, y, 10, 0, 2 * Math.PI);
    state.ctx.fill();
}

/**
 * 4. UI UPDATES
 */
function updateUI() {
    document.getElementById('val-velocity').textContent = state.currentVelocity.toFixed(2);
    document.getElementById('val-reps').textContent = state.repCount;
}

function setupVideoControls() {
    const video = state.video;
    const playBtn = document.getElementById('btn-play-pause');
    const seekBar = document.getElementById('seek-bar');
    const timeDisplay = document.getElementById('time-display');

    playBtn.onclick = (e) => {
        e.preventDefault();
        if (video.paused) {
            video.play();
            playBtn.textContent = "⏸";
        } else {
            video.pause();
            playBtn.textContent = "▶";
        }
    };

    seekBar.oninput = (e) => {
        video.currentTime = e.target.value;
    };

    video.ontimeupdate = () => {
        if (!isNaN(video.duration)) {
            seekBar.max = video.duration;
            seekBar.value = video.currentTime;
            const mins = Math.floor(video.currentTime / 60);
            const secs = Math.floor(video.currentTime % 60).toString().padStart(2, '0');
            timeDisplay.textContent = `${mins}:${secs}`;
        }
    };
}

// Start
initializeApp();
