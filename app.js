/**
 * PHASE 2 FIX: INFRASTRUCTURE
 * - Hides loading screen when ready
 * - Enables controls for uploaded videos
 * - Fixes looping issue
 */

import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const CONFIG = {
    VIDEO_WIDTH: 640,
    VIDEO_HEIGHT: 480,
    LANDMARK_WRIST: 16
};

let state = {
    video: null,
    canvas: null,
    ctx: null,
    landmarker: null,
    isReady: false,
    lastFrameTime: 0
};

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

        // 3. SUCCESS: Hide Loading Overlay
        console.log("✅ MediaPipe Ready");
        document.getElementById('loading-overlay').classList.add('hidden');
        document.getElementById('status-pill').textContent = "System Ready - Select Source";

        // 4. Attach Listeners
        document.getElementById('btn-camera').addEventListener('click', startCameraMode);
        document.getElementById('video-upload').addEventListener('change', startUploadMode);

    } catch (error) {
        console.error(error);
        alert("Startup Error: " + error.message);
    }
}

// --- CAMERA MODE ---
async function startCameraMode() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: CONFIG.VIDEO_WIDTH, height: CONFIG.VIDEO_HEIGHT, facingMode: 'user' } 
        });
        state.video.srcObject = stream;
        state.video.src = "";
        
        // Camera settings: No controls, mirrored usually preferred
        state.video.controls = false;
        state.video.loop = false;
        
        await playVideo();
    } catch (e) {
        alert("Camera Error: " + e.message);
    }
}

// --- UPLOAD MODE ---
function startUploadMode(event) {
    const file = event.target.files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    state.video.srcObject = null;
    state.video.src = url;
    
    // Upload settings: Enable controls, Disable loop
    state.video.controls = false; 
    state.video.loop = false; 
    
    playVideo();
}
function setupVideoControls() {
    const playBtn = document.getElementById('btn-play-pause');
    const seekBar = document.getElementById('seek-bar');
    const timeDisplay = document.getElementById('time-display');
    const video = state.video;

    // 1. Play/Pause Toggle
    playBtn.addEventListener('click', () => {
        if (video.paused) {
            video.play();
            playBtn.textContent = "⏸";
        } else {
            video.pause();
            playBtn.textContent = "▶";
        }
    });

    // 2. Seek Bar (Input)
    seekBar.addEventListener('input', () => {
        video.currentTime = seekBar.value;
    });

    // 3. Update Bar while video plays
    video.addEventListener('timeupdate', () => {
        // Update slider position
        seekBar.max = video.duration;
        seekBar.value = video.currentTime;
        
        // Update text (0:00)
        const mins = Math.floor(video.currentTime / 60);
        const secs = Math.floor(video.currentTime % 60).toString().padStart(2, '0');
        timeDisplay.textContent = `${mins}:${secs}`;
    });
}

// CALL THIS AT THE BOTTOM OF initializeApp()
setupVideoControls();

// --- SHARED STARTUP ---
async function playVideo() {
    return new Promise((resolve) => {
        state.video.onloadeddata = () => {
            // Match canvas size to video size
            state.canvas.width = state.video.videoWidth;
            state.canvas.height = state.video.videoHeight;
            
            state.video.play();
            if (!state.isReady) {
                state.isReady = true;
                window.requestAnimationFrame(renderLoop);
            }
            resolve();
        };
    });
}

// --- RENDER LOOP ---
function renderLoop(timestamp) {
    if (!state.isReady) return;

    if (state.video.paused || state.video.ended) {
        window.requestAnimationFrame(renderLoop);
        return;
    }

    if (state.video.currentTime !== state.lastFrameTime) {
        let startTimeMs = performance.now();
        if (state.video.duration) {
             startTimeMs = state.video.currentTime * 1000;
        }

        const result = state.landmarker.detectForVideo(state.video, startTimeMs);
        
        state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
        
        if (result.landmarks && result.landmarks.length > 0) {
            const wrist = result.landmarks[0][CONFIG.LANDMARK_WRIST];
            if (wrist) {
                // Draw Red Dot
                state.ctx.fillStyle = "red";
                state.ctx.beginPath();
                // Map normalized coordinates (0-1) to canvas size
                state.ctx.arc(wrist.x * state.canvas.width, wrist.y * state.canvas.height, 10, 0, 2 * Math.PI);
                state.ctx.fill();
            }
        }
        state.lastFrameTime = state.video.currentTime;
    }
    
    window.requestAnimationFrame(renderLoop);
}

initializeApp();
async function initializeApp() {
    // ... all the mediapipe loading code ...

    // AT THE VERY END OF THIS FUNCTION:
    setupVideoControls(); 
}

