import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const CONFIG = {
    WRIST_INDEX: 16,
    CONFIDENCE: 0.5,
    SMOOTHING: 3
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
    velocityBuffer: [],
    currentVel: 0
};

async function initializeApp() {
    try {
        state.video = document.getElementById('video');
        state.canvas = document.getElementById('canvas');
        state.ctx = state.canvas.getContext('2d');

        console.log("🚀 Starting App...");

        // 1. Load AI
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
            minPoseDetectionConfidence: CONFIG.CONFIDENCE
        });

        console.log("✅ AI Model Loaded");
        state.isModelLoaded = true;
        document.getElementById('loading-overlay').classList.add('hidden');
        document.getElementById('status-pill').textContent = "Select Video Source";

        // 2. Setup Inputs
        document.getElementById('btn-camera').onclick = startCamera;
        document.getElementById('file-input').onchange = handleUpload;
        
        // 3. Setup Video Events
        state.video.addEventListener('loadeddata', onVideoReady);
        state.video.addEventListener('error', (e) => alert("Video Error: " + state.video.error.message));

        // 4. Setup Test Controls
        const startBtn = document.getElementById('btn-start-test');
        const resetBtn = document.getElementById('btn-reset');
        const controlsDiv = document.querySelector('.video-controls');

        startBtn.onclick = () => {
            console.log("▶ Starting Test...");
            state.isTestRunning = true;
            state.prevWrist = null;
            state.velocityBuffer = [];
            
            startBtn.textContent = "Test Running...";
            startBtn.disabled = true;
            resetBtn.disabled = false;
            
            // Hide scrubber during test
            controlsDiv.style.display = 'none';
            
            if (state.video.paused) {
                state.video.play().catch(e => {
                    console.error("Play failed:", e);
                    alert("Tap video to start (Auto-play blocked)");
                });
            }
        };

        resetBtn.onclick = () => {
            console.log("↺ Resetting Test...");
            state.isTestRunning = false;
            state.prevWrist = null;
            state.ctx.clearRect(0,0,state.canvas.width, state.canvas.height);
            
            startBtn.textContent = "▶ Start Test";
            startBtn.disabled = false;
            resetBtn.disabled = true;
            document.getElementById('val-velocity').textContent = "0.00";
            
            // Show scrubber again
            controlsDiv.style.display = 'flex';
            
            state.video.pause();
            state.video.currentTime = 0;
        };
        
        setupVideoControls();

        // 5. Start Loop
        requestAnimationFrame(renderLoop);

    } catch (e) {
        alert("Startup Error: " + e.message);
    }
}

function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    console.log("📂 Loading file:", file.name);
    state.isVideoReady = false;
    document.getElementById('status-pill').textContent = "Loading File...";
    
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
    console.log("✅ Video Data Ready");
    state.isVideoReady = true;
    
    // Sync Canvas
    state.canvas.width = state.video.videoWidth;
    state.canvas.height = state.video.videoHeight;
    
    document.getElementById('status-pill').textContent = "Scanning for Wrist...";
    document.getElementById('btn-start-test').disabled = false;
    
    // Wake up mobile rendering
    state.video.play().then(() => {
        setTimeout(() => {
            if(!state.isTestRunning) state.video.pause();
        }, 100);
    }).catch(e => console.log("Autoplay blocked"));
}

function renderLoop() {
    if (state.isModelLoaded && state.isVideoReady) {
        if (state.video.readyState >= 2) {
            let startTimeMs = performance.now();
            if (state.video.duration) startTimeMs = state.video.currentTime * 1000;

            const result = state.landmarker.detectForVideo(state.video, startTimeMs);
            state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
            
            if (result.landmarks && result.landmarks.length > 0) {
                const wrist = result.landmarks[0][CONFIG.WRIST_INDEX];
                if (wrist) {
                    drawDot(wrist);
                    document.getElementById('status-pill').textContent = "Wrist Found ✓";
                    document.getElementById('status-pill').style.color = "#10b981";
                    
                    if (state.isTestRunning) {
                        calculatePhysics(wrist, startTimeMs);
                    }
                }
            } else {
                 document.getElementById('status-pill').textContent = "Scanning...";
                 document.getElementById('status-pill').style.color = "#fbbf24";
            }
        }
    }
    requestAnimationFrame(renderLoop);
}

function calculatePhysics(wrist, time) {
    // 1. First frame check
    if (!state.prevWrist) {
        state.prevWrist = { x: wrist.x, y: wrist.y, time: time };
        return;
    }

    // 2. Time Delta Check (Prevent Stuck Dot)
    const dt = (time - state.prevWrist.time) / 1000;
    if (dt < 0.01) return; // Skip invalid frames

    const x = wrist.x * state.canvas.width;
    const y = wrist.y * state.canvas.height;
    
    const dx = x - (state.prevWrist.x * state.canvas.width);
    const dy = y - (state.prevWrist.y * state.canvas.height);
    const dist = Math.sqrt(dx*dx + dy*dy);
    
    // Calibration: 2m width
    const scale = state.canvas.width / 2.0;
    
    const vel = (dist / scale) / dt;
    
    // Smoothing
    state.velocityBuffer.push(vel);
    if (state.velocityBuffer.length > CONFIG.SMOOTHING) state.velocityBuffer.shift();
    const smoothed = state.velocityBuffer.reduce((a,b)=>a+b)/state.velocityBuffer.length;
    
    document.getElementById('val-velocity').textContent = smoothed.toFixed(2);
    
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

function setupVideoControls() {
    const btn = document.getElementById('btn-play-pause');
    const bar = document.getElementById('seek-bar');
    const display = document.getElementById('time-display');
    
    btn.onclick = (e) => {
        e.preventDefault();
        state.video.paused ? state.video.play() : state.video.pause();
    };
    bar.oninput = (e) => state.video.currentTime = e.target.value;
    state.video.ontimeupdate = () => {
        bar.max = state.video.duration;
        bar.value = state.video.currentTime;
        const mins = Math.floor(state.video.currentTime / 60);
        const secs = Math.floor(state.video.currentTime % 60).toString().padStart(2, '0');
        display.textContent = `${mins}:${secs}`;
    };
}

initializeApp();
