/**
 * STEP 1: CONFIGURATION & STATE
 * Centralized control for physics, thresholds, and app state.
 */

const CONFIG = {
    // PHYSICS & CALIBRATION
    // Assumes average shoulder width (0.4m) to calibrate pixels-to-meters
    SHOULDER_WIDTH_M: 0.4, 
    // 3-frame rolling average smoothens jitter without lag
    SMOOTHING_WINDOW: 3,   
    
    // THRESHOLDS (The "Coaching" Logic)
    VELOCITY_PEAK_MIN: 0.5,   // m/s (Minimum speed to count as a rep)
    VELOCITY_RESET_MAX: 0.2,  // m/s (Speed must drop below this to reset for next rep)
    
    // ANAEROBIC DROP-OFF ALERTS
    DROP_WARN: 0.15, // 15% drop (Yellow)
    DROP_FAIL: 0.20, // 20% drop (Red)
    DROP_STOP: 0.25, // 25% drop (Stop Test)
    
    // SYSTEM
    VIDEO_WIDTH: 640,
    VIDEO_HEIGHT: 480,
    LANDMARK_WRIST: 16, // MediaPipe index for Right Wrist
};

// Global State Object - The "Single Source of Truth"
let state = {
    // Infrastructure
    video: null,
    canvas: null,
    ctx: null,
    landmarker: null,
    isReady: false,
    
    // Runtime Physics
    lastFrameTime: 0,
    pixelScale: 1,      // px to meters multiplier
    velocityHistory: [], // For smoothing
    prevWristPos: null, // {x, y, time}
    
    // Workout Data
    isTestRunning: false,
    repCount: 0,
    baselineVelocity: 0, // Avg of first 3 reps
    repPeaks: [],       // Array of peak velocities per rep
    currentVel: 0,      // Real-time velocity
    
    // Logic Flags
    inRep: false,       // Are we currently in the middle of a rep?
    repPeakTemp: 0      // Tracking the peak of the CURRENT rep
};

// HELPER: Simple rolling average for smoothing data
function getRollingAverage(arr) {
    if (arr.length === 0) return 0;
    const sum = arr.reduce((a, b) => a + b, 0);
    return sum / arr.length;
}

console.log("✅ Phase 1: Config & State loaded");
/**
 * STEP 2 (REVISED): FLEXIBLE INFRASTRUCTURE
 * Handles both Webcam and Video File Uploads
 */

import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

async function initializeApp() {
    try {
        console.log("🚀 Starting initialization...");
        
        // 1. Setup DOM Elements
        state.video = document.getElementById('video');
        state.canvas = document.getElementById('canvas');
        state.ctx = state.canvas.getContext('2d');
        
        // 2. Initialize MediaPipe (Async load)
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
        
        console.log("✅ MediaPipe Loaded. Waiting for user input...");

        // 3. Attach Event Listeners
        document.getElementById('btn-camera').addEventListener('click', startCameraMode);
        document.getElementById('video-upload').addEventListener('change', startUploadMode);
        
    } catch (error) {
        console.error("❌ Initialization Failed:", error);
        alert("Startup Error: " + error.message);
    }
}

// --- PATH A: CAMERA MODE ---
async function startCameraMode() {
    try {
        state.isTestRunning = false; // Reset state
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: CONFIG.VIDEO_WIDTH, height: CONFIG.VIDEO_HEIGHT, facingMode: 'user' } 
        });
        state.video.srcObject = stream;
        state.video.src = ""; // Clear any blob URL if exists
        
        await playVideo();
        console.log("✅ Camera Mode Active");
    } catch (e) {
        alert("Camera Error: " + e.message);
    }
}

// --- PATH B: UPLOAD MODE ---
function startUploadMode(event) {
    const file = event.target.files[0];
    if (!file) return;

    state.isTestRunning = false; // Reset state
    const url = URL.createObjectURL(file);
    state.video.srcObject = null; // Clear camera stream
    state.video.src = url;
    
    // Looping feels natural for testing uploads, but you can remove 'loop' if preferred
    state.video.loop = true; 
    
    playVideo().then(() => {
        console.log("✅ Upload Mode Active: " + file.name);
    });
}

// --- SHARED: START EVERYTHING ---
async function playVideo() {
    return new Promise((resolve) => {
        state.video.onloadeddata = () => {
            state.video.play();
            // Start the loop only once
            if (!state.isReady) {
                state.isReady = true;
                window.requestAnimationFrame(renderLoop);
            }
            resolve();
        };
    });
}

// --- THE GAME LOOP (Source Agnostic) ---
function renderLoop(timestamp) {
    if (!state.isReady) return;

    // Use video time (current time) instead of real time for syncing
    // This ensures that if the video pauses, the processing pauses too
    if (state.video.paused || state.video.ended) {
        window.requestAnimationFrame(renderLoop);
        return;
    }

    // Process frame if the video has advanced
    // We check if the timestamp has changed enough to warrant a new detection
    if (state.video.currentTime !== state.lastFrameTime) {
        
        // Pass the timestamp (in ms) to MediaPipe
        let startTimeMs = performance.now();
        if (state.video.duration) {
             // For files, we use the video's current time converted to ms
             startTimeMs = state.video.currentTime * 1000;
        }

        const result = state.landmarker.detectForVideo(state.video, startTimeMs);
        
        // Clear Canvas
        state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
        
        // Debug Draw
        if (result.landmarks && result.landmarks.length > 0) {
            const landmarks = result.landmarks[0];
            const wrist = landmarks[CONFIG.LANDMARK_WRIST];
            
            if (wrist) {
                // Draw Red Dot
                state.ctx.fillStyle = "red";
                state.ctx.beginPath();
                state.ctx.arc(wrist.x * state.canvas.width, wrist.y * state.canvas.height, 10, 0, 2 * Math.PI);
                state.ctx.fill();
            }
        }
        
        state.lastFrameTime = state.video.currentTime;
    }
    
    window.requestAnimationFrame(renderLoop);
}

// Initialize
initializeApp();
