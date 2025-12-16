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
    isModelLoaded: false, // AI loaded
    isVideoReady: false,  // Video loaded
    isTestRunning: false,
    lastVideoTime: -1,
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
            numPoses: 1
        });

        console.log("✅ AI Model Loaded");
        state.isModelLoaded = true;
        document.getElementById('loading-overlay').classList.add('hidden');
        document.getElementById('status-pill').textContent = "Select Video Source";

        // 2. Setup Inputs
        document.getElementById('btn-camera').onclick = startCamera;
        document.getElementById('file-input').onchange = handleUpload;
        
        // 3. Setup Video Event Listeners (The Fix)
        state.video.addEventListener('loadeddata', onVideoReady);
        state.video.addEventListener('error', (e) => alert("Video Error: " + state.video.error.message));

        // 4. Setup Controls
        document.getElementById('btn-start-test').onclick = () => state.isTestRunning = true;
        document.getElementById('btn-reset').onclick = () => {
            state.isTestRunning = false;
            state.prevWrist = null;
            state.ctx.clearRect(0,0,state.canvas.width, state.canvas.height);
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
    state.video.load(); // Force load
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
    
    // Force a single play/pause to wake up mobile rendering
    state.video.play().then(() => {
        setTimeout(() => {
            if(!state.isTestRunning) state.video.pause();
        }, 100);
    }).catch(e => console.log("Autoplay blocked"));
}

function renderLoop() {
    // Only run if AI is loaded AND Video is ready
    if (state.isModelLoaded && state.isVideoReady) {
        
        // Run detection if video has data (playing or paused-but-loaded)
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
    const x = wrist.x * state.canvas.width;
    const y = wrist.y * state.canvas.height;
    
    if (state.prevWrist) {
        const dx = x - state.prevWrist.x;
        const dy = y - state.prevWrist.y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        const dt = (time - state.prevWrist.time) / 1000;
        
        // Calibration: 2m width
        const scale = state.canvas.width / 2.0;
        
        if (dt > 0.005) {
            const vel = (dist / scale) / dt;
            state.velocityBuffer.push(vel);
            if (state.velocityBuffer.length > CONFIG.SMOOTHING) state.velocityBuffer.shift();
            const smoothed = state.velocityBuffer.reduce((a,b)=>a+b)/state.velocityBuffer.length;
            
            document.getElementById('val-velocity').textContent = smoothed.toFixed(2);
        }
    }
    state.prevWrist = { x, y, time };
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
    
    btn.onclick = (e) => {
        e.preventDefault();
        state.video.paused ? state.video.play() : state.video.pause();
    };
    bar.oninput = (e) => state.video.currentTime = e.target.value;
    state.video.ontimeupdate = () => {
        bar.max = state.video.duration;
        bar.value = state.video.currentTime;
    };
}

initializeApp();
