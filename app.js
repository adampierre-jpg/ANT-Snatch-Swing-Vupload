import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

/**
 * CONFIGURATION
 */
const CONFIG = {
    WRIST_INDEX: 16,     // Right Wrist
    CONFIDENCE: 0.5,     // Minimum confidence to "Green Light"
    SMOOTHING: 3         // Frames to smooth velocity
};

let state = {
    // Infrastructure
    video: null,
    canvas: null,
    ctx: null,
    landmarker: null,
    
    // Status Flags
    isSystemReady: false,   // MediaPipe loaded
    isWristFound: false,    // Wrist detected in current frame
    isTestRunning: false,   // "Start Test" active
    
    // Physics Data
    lastVideoTime: -1,
    prevWrist: null,
    velocityBuffer: [],
    
    // Metrics
    currentVel: 0,
    repCount: 0
};

/**
 * 1. INITIALIZATION & SETUP
 */
async function initializeApp() {
    try {
        // DOM Setup
        state.video = document.getElementById('video');
        state.canvas = document.getElementById('canvas');
        state.ctx = state.canvas.getContext('2d');
        const statusPill = document.getElementById('status-pill');

        console.log("⏳ Initializing AI...");
        const visionGen = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        
        state.landmarker = await PoseLandmarker.createFromOptions(visionGen, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numPoses: 1,
            minPoseDetectionConfidence: CONFIG.CONFIDENCE,
            minPosePresenceConfidence: CONFIG.CONFIDENCE,
            minTrackingConfidence: CONFIG.CONFIDENCE
        });

        console.log("✅ AI System Ready");
        state.isSystemReady = true;
        document.getElementById('loading-overlay').classList.add('hidden');
        statusPill.textContent = "Waiting for Source...";
        statusPill.style.borderColor = "#fbbf24"; // Yellow

        // Wire Inputs
        document.getElementById('btn-camera').onclick = startCamera;
        document.getElementById('file-input').onchange = handleUpload;
        
        // Wire Controls
        document.getElementById('btn-start-test').onclick = startTest;
        document.getElementById('btn-reset').onclick = resetTest;
        
        // Setup Video UI
        setupVideoControls();

        // START THE RENDER LOOP (Always running, scanning for wrists)
        requestAnimationFrame(renderLoop);

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
        resetPhysics();
    } catch (e) {
        alert("Camera Error: " + e.message);
    }
}

function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    state.video.srcObject = null;
    state.video.src = URL.createObjectURL(file);
    resetPhysics();
}

/**
 * 3. MAIN RENDER LOOP (The Heartbeat)
 */
function renderLoop() {
    if (!state.isSystemReady) {
        requestAnimationFrame(renderLoop);
        return;
    }

    // 1. Sync Canvas Size (Robust Check)
    if (state.video.videoWidth > 0 && state.canvas.width !== state.video.videoWidth) {
        state.canvas.width = state.video.videoWidth;
        state.canvas.height = state.video.videoHeight;
        console.log(`Canvas Resized to: ${state.canvas.width}x${state.canvas.height}`);
    }

    // 2. Frame Detection Logic
    // We only detect if video is playing OR if it's paused but we just loaded it (for scanning)
    if (!state.video.paused || (state.video.paused && state.lastVideoTime !== state.video.currentTime)) {
        
        // Handle Video Loops/Seeks (Time moved backwards)
        if (state.video.currentTime < state.lastVideoTime) {
            console.log("↺ Time Jump Detected - Resetting Physics");
            state.prevWrist = null;
            state.velocityBuffer = [];
        }

        let startTimeMs = performance.now();
        if (state.video.duration) {
             startTimeMs = state.video.currentTime * 1000;
        }

        // EXECUTE AI
        const result = state.landmarker.detectForVideo(state.video, startTimeMs);
        
        // CLEAR CANVAS
        state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
        
        // PROCESS RESULTS
        if (result.landmarks && result.landmarks.length > 0) {
            const wrist = result.landmarks[0][CONFIG.WRIST_INDEX];
            
            if (wrist) {
                state.isWristFound = true;
                updateStatus(true);
                drawWrist(wrist);
                
                // Only calculate numbers if TEST IS RUNNING
                if (state.isTestRunning) {
                    calculatePhysics(wrist, startTimeMs);
                }
            } else {
                state.isWristFound = false;
                updateStatus(false);
            }
        } else {
            state.isWristFound = false;
            updateStatus(false);
        }
        
        state.lastVideoTime = state.video.currentTime;
    }
    
    requestAnimationFrame(renderLoop);
}

/**
 * 4. PHYSICS ENGINE
 */
function calculatePhysics(wrist, time) {
    const x = wrist.x * state.canvas.width;
    const y = wrist.y * state.canvas.height;
    
    if (state.prevWrist) {
        const dx = x - state.prevWrist.x;
        const dy = y - state.prevWrist.y;
        const distPx = Math.sqrt(dx*dx + dy*dy);
        
        // Time Delta (seconds)
        const dt = (time - state.prevWrist.time) / 1000;
        
        // Calibration: Assume width of view is ~2 meters (Standard gym shot)
        const pxPerMeter = state.canvas.width / 2.0;
        
        if (dt > 0.005) { // Avoid divide-by-zero on tiny steps
            const rawVel = (distPx / pxPerMeter) / dt;
            
            // Smoothing
            state.velocityBuffer.push(rawVel);
            if (state.velocityBuffer.length > CONFIG.SMOOTHING) state.velocityBuffer.shift();
            
            state.currentVel = state.velocityBuffer.reduce((a,b)=>a+b) / state.velocityBuffer.length;
            
            updateMetrics();
        }
    }
    
    state.prevWrist = { x, y, time };
}

function drawWrist(wrist) {
    const cx = wrist.x * state.canvas.width;
    const cy = wrist.y * state.canvas.height;
    
    // Outer Glow
    state.ctx.beginPath();
    state.ctx.arc(cx, cy, 15, 0, 2*Math.PI);
    state.ctx.fillStyle = "rgba(16, 185, 129, 0.3)"; // Green Glow
    state.ctx.fill();
    
    // Core Dot
    state.ctx.beginPath();
    state.ctx.arc(cx, cy, 6, 0, 2*Math.PI);
    state.ctx.fillStyle = "#10b981"; // Green-500
    state.ctx.fill();
}

/**
 * 5. UI & CONTROLS
 */
function startTest() {
    if (!state.isWristFound) {
        alert("⚠️ Cannot start: No wrist detected. Please adjust camera/video.");
        return;
    }
    state.isTestRunning = true;
    state.velocityBuffer = [];
    state.prevWrist = null; // Reset prev position to avoid huge velocity jump
    
    document.getElementById('btn-start-test').disabled = true;
    document.getElementById('btn-start-test').textContent = "Running...";
    document.getElementById('btn-reset').disabled = false;
    
    if (state.video.paused) state.video.play();
}

function resetTest() {
    state.isTestRunning = false;
    resetPhysics();
    document.getElementById('btn-start-test').disabled = false;
    document.getElementById('btn-start-test').textContent = "▶ Start Test";
    document.getElementById('val-velocity').textContent = "0.00";
    
    // Pause video to let user reset
    state.video.pause();
    state.video.currentTime = 0;
}

function resetPhysics() {
    state.prevWrist = null;
    state.velocityBuffer = [];
    state.currentVel = 0;
    state.lastVideoTime = -1;
}

function updateStatus(found) {
    const pill = document.getElementById('status-pill');
    if (found) {
        pill.textContent = "Wrist Detected ✓";
        pill.style.borderColor = "#10b981"; // Green
        pill.style.color = "#10b981";
    } else {
        pill.textContent = "Searching...";
        pill.style.borderColor = "#ef4444"; // Red
        pill.style.color = "#ef4444";
    }
}

function updateMetrics() {
    document.getElementById('val-velocity').textContent = state.currentVel.toFixed(2);
}

// VIDEO CONTROLS WIRING (Same as before)
function setupVideoControls() {
    const btn = document.getElementById('btn-play-pause');
    const bar = document.getElementById('seek-bar');
    
    btn.onclick = (e) => {
        e.preventDefault();
        state.video.paused ? state.video.play() : state.video.pause();
    };
    
    bar.oninput = (e) => { state.video.currentTime = e.target.value; };
    
    state.video.ontimeupdate = () => {
        bar.max = state.video.duration || 100;
        bar.value = state.video.currentTime;
        btn.textContent = state.video.paused ? "▶" : "⏸";
    };
}

// BOOTSTRAP
initializeApp();
