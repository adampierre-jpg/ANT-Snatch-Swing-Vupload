/**
 * Kettlebell Velocity Tracker - Main Application
 * Real-time velocity tracking using MediaPipe Pose for anaerobic threshold detection
 */

// Global State
const state = {
    poseLandmarker: null,
    isInitialized: false,
    isRunning: false,
    selectedExercise: null,
    selectedWrist: null,
    inputMode: null, // 'camera' or 'video'

    // Video elements
    video: null,
    canvas: null,
    canvasCtx: null,

    // Tracking data
    lastVideoTime: -1,
    previousPosition: null,
    previousTimestamp: null,
    velocityHistory: [],

    // Rep and velocity tracking
    currentVelocity: 0,
    peakVelocity: 0,
    baselineVelocity: null,
    repCount: 0,
    repPeakVelocities: [],
    velocityDropPercent: 0,

    // Rep detection
    isInRep: false,
    currentRepPeak: 0,
    velocityThreshold: 0.3, // m/s - minimum velocity to consider movement

    // Calibration (pixels to meters)
    pixelToMeterScale: 1,
    shoulderWidth: 0.4, // Average shoulder width in meters

    // Animation
    animationId: null,

    // Alerts
    alertedAt20: false,
    alertedAt25: false
};

// DOM Elements
const elements = {
    // Sections
    exerciseSelection: document.getElementById('exerciseSelection'),
    modeSelection: document.getElementById('modeSelection'),
    videoSection: document.getElementById('videoSection'),
    resultsSection: document.getElementById('resultsSection'),

    // Exercise buttons
    exerciseButtons: document.querySelectorAll('.exercise-btn'),

    // Mode buttons
    liveCameraBtn: document.getElementById('liveCameraBtn'),
    uploadVideoBtn: document.getElementById('uploadVideoBtn'),
    videoFileInput: document.getElementById('videoFileInput'),

    // Video/Canvas
    webcam: document.getElementById('webcam'),
    canvas: document.getElementById('canvas'),
    loadingOverlay: document.getElementById('loadingOverlay'),

    // Controls
    startBtn: document.getElementById('startBtn'),
    stopBtn: document.getElementById('stopBtn'),
    resetBtn: document.getElementById('resetBtn'),
    backBtn: document.getElementById('backBtn'),

    // Metrics
    currentVelocityEl: document.getElementById('currentVelocity'),
    peakVelocityEl: document.getElementById('peakVelocity'),
    baselineVelocityEl: document.getElementById('baselineVelocity'),
    velocityDropEl: document.getElementById('velocityDrop'),
    repCountEl: document.getElementById('repCount'),
    statusTextEl: document.getElementById('statusText'),
    statusCard: document.getElementById('statusCard'),

    // Alert
    thresholdAlert: document.getElementById('thresholdAlert'),
    alertMessage: document.getElementById('alertMessage'),

    // Results
    resultsContainer: document.getElementById('resultsContainer'),
    clearResultsBtn: document.getElementById('clearResultsBtn')
};

/**
 * Initialize the application
 */
async function init() {
    console.log('Initializing KB Velocity Tracker...');

    // Setup event listeners
    setupEventListeners();

    // Load previous results
    loadResults();

    console.log('Application initialized');
}

/**
 * Setup all event listeners
 */
function setupEventListeners() {
    // Exercise selection
    elements.exerciseButtons.forEach(btn => {
        btn.addEventListener('click', () => selectExercise(btn));
    });

    // Mode selection
    elements.liveCameraBtn.addEventListener('click', () => selectMode('camera'));
    elements.uploadVideoBtn.addEventListener('click', () => {
        elements.videoFileInput.click();
    });

    elements.videoFileInput.addEventListener('change', handleVideoUpload);

    // Control buttons
    elements.startBtn.addEventListener('click', startTest);
    elements.stopBtn.addEventListener('click', stopTest);
    elements.resetBtn.addEventListener('click', resetTest);
    elements.backBtn.addEventListener('click', backToSelection);

    // Results
    elements.clearResultsBtn.addEventListener('click', clearAllResults);
}

/**
 * Select exercise type
 */
function selectExercise(button) {
    // Remove previous selection
    elements.exerciseButtons.forEach(btn => btn.classList.remove('selected'));

    // Select new exercise
    button.classList.add('selected');
    state.selectedExercise = button.dataset.exercise;
    state.selectedWrist = button.dataset.wrist;

    console.log(`Exercise selected: ${state.selectedExercise}, Wrist: ${state.selectedWrist}`);

    // Show mode selection
    elements.exerciseSelection.style.display = 'none';
    elements.modeSelection.style.display = 'block';
}

/**
 * Select input mode (camera or video upload)
 */
async function selectMode(mode) {
    state.inputMode = mode;
    console.log(`Input mode selected: ${mode}`);

    // Show video section
    elements.modeSelection.style.display = 'none';
    elements.videoSection.style.display = 'block';

    if (mode === 'camera') {
        await initializeCamera();
    }
}

/**
 * Handle video file upload
 */
function handleVideoUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    console.log(`Video file selected: ${file.name}`);

    state.inputMode = 'video';

    // Show video section
    elements.modeSelection.style.display = 'none';
    elements.videoSection.style.display = 'block';

    // Load video file
    const url = URL.createObjectURL(file);
    elements.webcam.src = url;
    elements.webcam.muted = true;
    elements.webcam.loop = false;

    // Initialize MediaPipe when video is loaded
    elements.webcam.addEventListener('loadedmetadata', async () => {
        console.log(`Video loaded: ${elements.webcam.videoWidth}x${elements.webcam.videoHeight}, duration: ${elements.webcam.duration}s`);
        await initializeMediaPipe();
    }, { once: true });
}

/**
 * Initialize camera access
 */
async function initializeCamera() {
    try {
        console.log('Requesting camera access...');

        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user',
                frameRate: { ideal: 30 }
            }
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        elements.webcam.srcObject = stream;

        // Wait for video to be ready
        await new Promise((resolve) => {
            elements.webcam.onloadedmetadata = () => {
                console.log(`Camera initialized: ${elements.webcam.videoWidth}x${elements.webcam.videoHeight}`);
                resolve();
            };
        });

        await initializeMediaPipe();

    } catch (error) {
        console.error('Camera access error:', error);
        showError(`Camera access denied or not available: ${error.message}`);
    }
}

/**
 * Initialize MediaPipe Pose Landmarker
 */
async function initializeMediaPipe() {
    if (state.isInitialized) {
        console.log('MediaPipe already initialized');
        hideLoadingOverlay();
        return;
    }

    try {
        console.log('Initializing MediaPipe Pose Landmarker...');
        showLoadingOverlay('Initializing MediaPipe...');

        // Wait for MediaPipe library to load
        if (typeof vision === 'undefined') {
            await new Promise((resolve) => {
                const checkLibrary = setInterval(() => {
                    if (typeof vision !== 'undefined') {
                        clearInterval(checkLibrary);
                        resolve();
                    }
                }, 100);
            });
        }

        const { PoseLandmarker, FilesetResolver } = vision;

        // Load model files
        const filesetResolver = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        // Create pose landmarker with optimized settings
        state.poseLandmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numPoses: 1,
            minPoseDetectionConfidence: 0.5,
            minPosePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        console.log('MediaPipe Pose Landmarker initialized successfully');

        // Setup canvas
        setupCanvas();

        state.isInitialized = true;
        hideLoadingOverlay();

        // Enable start button
        elements.startBtn.disabled = false;
        updateStatus('Ready to start');

    } catch (error) {
        console.error('MediaPipe initialization error:', error);
        showError(`Failed to initialize MediaPipe: ${error.message}`);
        hideLoadingOverlay();
    }
}

/**
 * Setup canvas for drawing
 */
function setupCanvas() {
    state.video = elements.webcam;
    state.canvas = elements.canvas;
    state.canvasCtx = state.canvas.getContext('2d');

    // Set canvas size to match video
    state.canvas.width = state.video.videoWidth;
    state.canvas.height = state.video.videoHeight;

    console.log(`Canvas setup: ${state.canvas.width}x${state.canvas.height}`);
}

/**
 * Start the velocity test
 */
function startTest() {
    if (!state.isInitialized) {
        showError('MediaPipe not initialized. Please wait.');
        return;
    }

    console.log('Starting velocity test...');

    state.isRunning = true;
    resetMetrics();

    // Update UI
    elements.startBtn.disabled = true;
    elements.stopBtn.disabled = false;
    updateStatus('Testing...');

    // Start video if uploaded
    if (state.inputMode === 'video') {
        state.video.play();
    }

    // Start processing frames
    state.lastVideoTime = -1;
    processFrame();
}

/**
 * Stop the velocity test
 */
function stopTest() {
    console.log('Stopping velocity test...');

    state.isRunning = false;

    // Stop animation
    if (state.animationId) {
        cancelAnimationFrame(state.animationId);
        state.animationId = null;
    }

    // Pause video if uploaded
    if (state.inputMode === 'video') {
        state.video.pause();
    }

    // Update UI
    elements.startBtn.disabled = false;
    elements.stopBtn.disabled = true;
    updateStatus('Test stopped');

    // Save results if we have data
    if (state.repCount > 0) {
        saveResults();
    }
}

/**
 * Reset test metrics
 */
function resetMetrics() {
    state.previousPosition = null;
    state.previousTimestamp = null;
    state.velocityHistory = [];
    state.currentVelocity = 0;
    state.peakVelocity = 0;
    state.baselineVelocity = null;
    state.repCount = 0;
    state.repPeakVelocities = [];
    state.velocityDropPercent = 0;
    state.isInRep = false;
    state.currentRepPeak = 0;
    state.alertedAt20 = false;
    state.alertedAt25 = false;

    updateUI();
    hideAlert();
}

/**
 * Reset entire test
 */
function resetTest() {
    console.log('Resetting test...');

    stopTest();
    resetMetrics();

    // Clear canvas
    if (state.canvasCtx) {
        state.canvasCtx.clearRect(0, 0, state.canvas.width, state.canvas.height);
    }

    // Reset video
    if (state.inputMode === 'video') {
        state.video.currentTime = 0;
    }

    updateStatus('Ready to start');
}

/**
 * Back to exercise selection
 */
function backToSelection() {
    // Stop everything
    stopTest();

    // Stop camera if active
    if (state.inputMode === 'camera' && state.video.srcObject) {
        state.video.srcObject.getTracks().forEach(track => track.stop());
        state.video.srcObject = null;
    }

    // Reset state
    state.isInitialized = false;
    state.selectedExercise = null;
    state.selectedWrist = null;
    state.inputMode = null;
    state.poseLandmarker = null;

    // Clear file input
    elements.videoFileInput.value = '';

    // Show exercise selection
    elements.videoSection.style.display = 'none';
    elements.modeSelection.style.display = 'none';
    elements.exerciseSelection.style.display = 'block';

    // Remove exercise selection
    elements.exerciseButtons.forEach(btn => btn.classList.remove('selected'));

    console.log('Returned to exercise selection');
}

/**
 * Process each video frame
 */
function processFrame() {
    if (!state.isRunning) return;

    const currentTime = performance.now();

    // Check if this is a new frame (avoid processing same frame twice)
    if (state.video.currentTime !== state.lastVideoTime) {
        state.lastVideoTime = state.video.currentTime;

        // Detect pose
        const results = state.poseLandmarker.detectForVideo(state.video, currentTime);

        // Process results
        if (results && results.landmarks && results.landmarks.length > 0) {
            processpose(results.landmarks[0], currentTime);
            drawResults(results);
        } else {
            // Clear canvas if no pose detected
            state.canvasCtx.clearRect(0, 0, state.canvas.width, state.canvas.height);
            drawNoPoiseDetected();
        }

        updateUI();
    }

    // Continue processing
    state.animationId = requestAnimationFrame(processFrame);
}

/**
 * Process pose landmarks to calculate velocity
 */
function processpose(landmarks, timestamp) {
    // Get wrist position(s)
    let wristPosition = null;

    if (state.selectedWrist === 'both') {
        // Average both wrists for two-arm exercises
        const leftWrist = landmarks[15];  // Left wrist
        const rightWrist = landmarks[16]; // Right wrist
        wristPosition = {
            x: (leftWrist.x + rightWrist.x) / 2,
            y: (leftWrist.y + rightWrist.y) / 2
        };
    } else {
        // Single wrist (15 = left, 16 = right)
        const wristIndex = parseInt(state.selectedWrist);
        wristPosition = landmarks[wristIndex];
    }

    // Calibrate pixel-to-meter scale on first frame
    if (!state.previousPosition) {
        calibrateScale(landmarks);
    }

    // Calculate velocity
    if (state.previousPosition && state.previousTimestamp) {
        const timeDelta = (timestamp - state.previousTimestamp) / 1000; // Convert to seconds

        if (timeDelta > 0) {
            // Calculate distance in pixels
            const dx = (wristPosition.x - state.previousPosition.x) * state.canvas.width;
            const dy = (wristPosition.y - state.previousPosition.y) * state.canvas.height;
            const distancePixels = Math.sqrt(dx * dx + dy * dy);

            // Convert to meters
            const distanceMeters = distancePixels * state.pixelToMeterScale;

            // Calculate velocity (m/s)
            const instantVelocity = distanceMeters / timeDelta;

            // Add to history for smoothing
            state.velocityHistory.push(instantVelocity);
            if (state.velocityHistory.length > 3) {
                state.velocityHistory.shift(); // Keep only last 3 values
            }

            // Apply 3-frame rolling average
            state.currentVelocity = state.velocityHistory.reduce((a, b) => a + b, 0) / state.velocityHistory.length;

            // Track peak velocity
            if (state.currentVelocity > state.peakVelocity) {
                state.peakVelocity = state.currentVelocity;
            }

            // Detect reps
            detectRep();
        }
    }

    // Update previous position and timestamp
    state.previousPosition = { x: wristPosition.x, y: wristPosition.y };
    state.previousTimestamp = timestamp;
}

/**
 * Calibrate pixel-to-meter scale using shoulder width
 */
function calibrateScale(landmarks) {
    const leftShoulder = landmarks[11];
    const rightShoulder = landmarks[12];

    const shoulderWidthPixels = Math.sqrt(
        Math.pow((rightShoulder.x - leftShoulder.x) * state.canvas.width, 2) +
        Math.pow((rightShoulder.y - leftShoulder.y) * state.canvas.height, 2)
    );

    // Calculate scale factor
    state.pixelToMeterScale = state.shoulderWidth / shoulderWidthPixels;

    console.log(`Calibrated scale: ${state.pixelToMeterScale.toFixed(6)} m/pixel (shoulder width: ${shoulderWidthPixels.toFixed(2)}px)`);
}

/**
 * Detect rep based on velocity patterns
 */
function detectRep() {
    const velocity = state.currentVelocity;

    // Start of rep: velocity crosses threshold going up
    if (!state.isInRep && velocity > state.velocityThreshold) {
        state.isInRep = true;
        state.currentRepPeak = velocity;
        console.log(`Rep started at velocity: ${velocity.toFixed(2)} m/s`);
    }

    // During rep: track peak
    if (state.isInRep) {
        if (velocity > state.currentRepPeak) {
            state.currentRepPeak = velocity;
        }

        // End of rep: velocity drops below threshold
        if (velocity < state.velocityThreshold * 0.5) {
            completeRep();
        }
    }
}

/**
 * Complete a rep and update metrics
 */
function completeRep() {
    state.isInRep = false;
    state.repCount++;

    // Store peak velocity for this rep
    state.repPeakVelocities.push(state.currentRepPeak);

    console.log(`Rep ${state.repCount} completed. Peak velocity: ${state.currentRepPeak.toFixed(2)} m/s`);

    // Calculate baseline after 2-3 reps
    if (state.repCount >= 2 && state.baselineVelocity === null) {
        calculateBaseline();
    }

    // Check threshold if baseline is set
    if (state.baselineVelocity !== null) {
        checkThreshold();
    }

    // Reset current rep peak
    state.currentRepPeak = 0;
}

/**
 * Calculate baseline velocity from first 2-3 reps
 */
function calculateBaseline() {
    const repsToAverage = Math.min(3, state.repPeakVelocities.length);
    const sum = state.repPeakVelocities.slice(0, repsToAverage).reduce((a, b) => a + b, 0);
    state.baselineVelocity = sum / repsToAverage;

    console.log(`Baseline calculated: ${state.baselineVelocity.toFixed(2)} m/s (from ${repsToAverage} reps)`);
    updateStatus('Baseline set');
}

/**
 * Check velocity drop threshold
 */
function checkThreshold() {
    const latestPeak = state.repPeakVelocities[state.repPeakVelocities.length - 1];

    // Calculate velocity drop percentage
    state.velocityDropPercent = ((state.baselineVelocity - latestPeak) / state.baselineVelocity) * 100;

    // Ensure it doesn't go negative
    if (state.velocityDropPercent < 0) {
        state.velocityDropPercent = 0;
    }

    console.log(`Velocity drop: ${state.velocityDropPercent.toFixed(1)}%`);

    // Alert at 20% threshold
    if (state.velocityDropPercent >= 20 && !state.alertedAt20) {
        state.alertedAt20 = true;
        showAlert('20% velocity drop detected. Approaching anaerobic threshold.');
        updateStatus('Warning: 20% drop');
    }

    // Alert at 25% threshold (critical)
    if (state.velocityDropPercent >= 25 && !state.alertedAt25) {
        state.alertedAt25 = true;
        showAlert('25% velocity drop! Anaerobic threshold reached. Consider stopping test.');
        updateStatus('Critical: 25% drop');
    }
}

/**
 * Draw pose landmarks and skeleton on canvas
 */
function drawResults(results) {
    const ctx = state.canvasCtx;
    ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);

    if (results.landmarks && results.landmarks.length > 0) {
        const landmarks = results.landmarks[0];

        // Draw connections (skeleton)
        drawConnections(ctx, landmarks);

        // Draw landmarks (keypoints)
        drawLandmarks(ctx, landmarks);

        // Highlight tracked wrist(s)
        highlightTrackedWrist(ctx, landmarks);
    }
}

/**
 * Draw skeleton connections
 */
function drawConnections(ctx, landmarks) {
    const connections = [
        // Torso
        [11, 12], [11, 23], [12, 24], [23, 24],
        // Left arm
        [11, 13], [13, 15],
        // Right arm
        [12, 14], [14, 16],
        // Left leg
        [23, 25], [25, 27],
        // Right leg
        [24, 26], [26, 28]
    ];

    ctx.strokeStyle = 'rgba(0, 173, 181, 0.6)';
    ctx.lineWidth = 2;

    connections.forEach(([start, end]) => {
        const startPoint = landmarks[start];
        const endPoint = landmarks[end];

        if (startPoint && endPoint) {
            ctx.beginPath();
            ctx.moveTo(startPoint.x * state.canvas.width, startPoint.y * state.canvas.height);
            ctx.lineTo(endPoint.x * state.canvas.width, endPoint.y * state.canvas.height);
            ctx.stroke();
        }
    });
}

/**
 * Draw pose landmarks
 */
function drawLandmarks(ctx, landmarks) {
    ctx.fillStyle = 'rgba(0, 255, 136, 0.8)';

    landmarks.forEach((landmark, index) => {
        // Only draw visible landmarks
        if (landmark.visibility && landmark.visibility > 0.5) {
            ctx.beginPath();
            ctx.arc(
                landmark.x * state.canvas.width,
                landmark.y * state.canvas.height,
                5,
                0,
                2 * Math.PI
            );
            ctx.fill();
        }
    });
}

/**
 * Highlight the tracked wrist(s)
 */
function highlightTrackedWrist(ctx, landmarks) {
    const wristIndices = state.selectedWrist === 'both' ? [15, 16] : [parseInt(state.selectedWrist)];

    wristIndices.forEach(index => {
        const wrist = landmarks[index];
        if (wrist) {
            // Draw larger circle
            ctx.strokeStyle = '#ff0000';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(
                wrist.x * state.canvas.width,
                wrist.y * state.canvas.height,
                10,
                0,
                2 * Math.PI
            );
            ctx.stroke();

            // Draw velocity indicator
            const velocityBarHeight = Math.min(state.currentVelocity * 50, 100);
            ctx.fillStyle = 'rgba(255, 0, 0, 0.6)';
            ctx.fillRect(
                wrist.x * state.canvas.width - 15,
                wrist.y * state.canvas.height - velocityBarHeight - 20,
                10,
                velocityBarHeight
            );
        }
    });
}

/**
 * Draw "No Pose Detected" message
 */
function drawNoPoseDetected() {
    const ctx = state.canvasCtx;
    ctx.fillStyle = 'rgba(244, 67, 54, 0.8)';
    ctx.font = '20px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('No pose detected', state.canvas.width / 2, state.canvas.height / 2);
}

/**
 * Update UI with current metrics
 */
function updateUI() {
    elements.currentVelocityEl.textContent = state.currentVelocity.toFixed(2);
    elements.peakVelocityEl.textContent = state.peakVelocity.toFixed(2);
    elements.baselineVelocityEl.textContent = state.baselineVelocity !== null
        ? state.baselineVelocity.toFixed(2)
        : '--';
    elements.velocityDropEl.textContent = state.velocityDropPercent.toFixed(1);
    elements.repCountEl.textContent = state.repCount;

    // Update status card colors
    updateStatusColors();
}

/**
 * Update status card colors based on velocity drop
 */
function updateStatusColors() {
    const card = elements.statusCard;

    // Remove all status classes
    card.classList.remove('status-optimal', 'status-warning', 'status-critical');

    if (state.baselineVelocity === null) {
        // No baseline yet
        return;
    }

    if (state.velocityDropPercent < 15) {
        card.classList.add('status-optimal');
    } else if (state.velocityDropPercent < 20) {
        card.classList.add('status-warning');
    } else {
        card.classList.add('status-critical');
    }
}

/**
 * Update status text
 */
function updateStatus(text) {
    elements.statusTextEl.textContent = text;
}

/**
 * Show threshold alert
 */
function showAlert(message) {
    elements.alertMessage.textContent = message;
    elements.thresholdAlert.style.display = 'block';

    // Auto-hide after 5 seconds
    setTimeout(() => {
        hideAlert();
    }, 5000);
}

/**
 * Hide threshold alert
 */
function hideAlert() {
    elements.thresholdAlert.style.display = 'none';
}

/**
 * Show loading overlay
 */
function showLoadingOverlay(message = 'Loading...') {
    elements.loadingOverlay.style.display = 'flex';
    elements.loadingOverlay.querySelector('p').textContent = message;
}

/**
 * Hide loading overlay
 */
function hideLoadingOverlay() {
    elements.loadingOverlay.style.display = 'none';
}

/**
 * Show error message
 */
function showError(message) {
    alert(`Error: ${message}`);
    console.error(message);
}

/**
 * Save test results to localStorage
 */
function saveResults() {
    const result = {
        date: new Date().toISOString(),
        exercise: state.selectedExercise,
        inputMode: state.inputMode,
        repCount: state.repCount,
        baselineVelocity: state.baselineVelocity,
        finalPeakVelocity: state.repPeakVelocities[state.repPeakVelocities.length - 1] || 0,
        maxVelocityDrop: state.velocityDropPercent,
        peakVelocity: state.peakVelocity,
        allRepVelocities: state.repPeakVelocities
    };

    // Get existing results
    const results = JSON.parse(localStorage.getItem('kbVelocityResults') || '[]');

    // Add new result
    results.unshift(result); // Add to beginning

    // Keep only last 50 results
    if (results.length > 50) {
        results.length = 50;
    }

    // Save to localStorage
    localStorage.setItem('kbVelocityResults', JSON.stringify(results));

    console.log('Results saved:', result);

    // Reload results display
    loadResults();
}

/**
 * Load and display results from localStorage
 */
function loadResults() {
    const results = JSON.parse(localStorage.getItem('kbVelocityResults') || '[]');

    if (results.length === 0) {
        elements.resultsContainer.innerHTML = '<p class="no-results">No test results yet. Complete a test to see results here.</p>';
        elements.clearResultsBtn.style.display = 'none';
        return;
    }

    // Display results
    let html = '';
    results.forEach((result, index) => {
        const date = new Date(result.date);
        const formattedDate = date.toLocaleString();

        html += `
            <div class="result-item">
                <div class="result-header">
                    <div class="result-date">${formattedDate}</div>
                    <div class="result-exercise">${formatExerciseName(result.exercise)} (${result.inputMode === 'camera' ? 'Live' : 'Video'})</div>
                </div>
                <div class="result-stats">
                    <div class="result-stat">
                        <div class="result-stat-label">Reps</div>
                        <div class="result-stat-value">${result.repCount}</div>
                    </div>
                    <div class="result-stat">
                        <div class="result-stat-label">Baseline</div>
                        <div class="result-stat-value">${result.baselineVelocity ? result.baselineVelocity.toFixed(2) : '--'} m/s</div>
                    </div>
                    <div class="result-stat">
                        <div class="result-stat-label">Final Peak</div>
                        <div class="result-stat-value">${result.finalPeakVelocity.toFixed(2)} m/s</div>
                    </div>
                    <div class="result-stat">
                        <div class="result-stat-label">Max Drop</div>
                        <div class="result-stat-value">${result.maxVelocityDrop.toFixed(1)}%</div>
                    </div>
                    <div class="result-stat">
                        <div class="result-stat-label">Peak Velocity</div>
                        <div class="result-stat-value">${result.peakVelocity.toFixed(2)} m/s</div>
                    </div>
                </div>
            </div>
        `;
    });

    elements.resultsContainer.innerHTML = html;
    elements.clearResultsBtn.style.display = 'block';
}

/**
 * Format exercise name for display
 */
function formatExerciseName(exercise) {
    return exercise
        .split('-')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

/**
 * Clear all results from localStorage
 */
function clearAllResults() {
    if (confirm('Are you sure you want to clear all test results? This cannot be undone.')) {
        localStorage.removeItem('kbVelocityResults');
        loadResults();
        console.log('All results cleared');
    }
}

// Initialize application when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Fix typo in function name
function processPose(landmarks, timestamp) {
    processpose(landmarks, timestamp);
}
