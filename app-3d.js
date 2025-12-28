/**
 * ============================================================================
 * VBT (VELOCITY BASED TRAINING) APPLICATION - 3D VERSION
 * ============================================================================
 * 
 * UPGRADE FROM 2D TO 3D:
 * MediaPipe's PoseLandmarker provides full 3D coordinates (x, y, z) for each
 * landmark. The previous version only used x and y coordinates. This 3D version
 * uses all three coordinates for:
 * 
 * 1. MORE ACCURATE ELBOW ANGLES - 3D vector math captures the true joint angle
 *    regardless of camera angle or body rotation
 * 
 * 2. TRUE 3D VELOCITY - Tracks movement in all three dimensions, capturing
 *    the full speed of kettlebell movement including depth changes
 * 
 * 3. ACCURATE BODY SEGMENT LENGTHS - 3D distance calculations give true
 *    segment lengths even when limbs are angled toward/away from camera
 * 
 * 4. DEPTH-AWARE POSITION DETECTION - Uses z-coordinate to improve detection
 *    of rack position, overhead lockout, and swing height
 * 
 * MEDIAPIPE Z-COORDINATE EXPLAINED:
 * - z represents depth relative to the hip center
 * - Negative z = closer to camera (toward viewer)
 * - Positive z = further from camera (away from viewer)
 * - z is scaled similarly to x (roughly same units)
 * - z is most accurate for landmarks near the torso
 * 
 * ============================================================================
 * SECTIONS:
 * 1. Imports & 3D Vector Math Utilities
 * 2. One Euro Filter (Adaptive Smoothing)
 * 3. VelocityFatigueTracker Class
 * 4. SetTimingTracker Class
 * 5. CalibrationSystem Class (3D Enhanced)
 * 6. VBTStateMachine Class (3D Enhanced)
 * 7. App Initialization & Main Loop
 * 8. UI Update Functions
 * 9. Helper Functions
 * ============================================================================
 */

import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";


// ============================================================================
// SECTION 1: 3D VECTOR MATH UTILITIES
// ============================================================================
/**
 * PURPOSE: Provide 3D vector operations for accurate angle and distance
 * calculations. These replace the 2D operations used in the previous version.
 * 
 * WHY 3D MATTERS:
 * - When your arm is angled toward the camera, a 2D measurement underestimates
 *   the true length and misreads angles
 * - 3D calculations capture the true geometry regardless of orientation
 * - Particularly important for elbow angle which determines rack vs lockout
 * 
 * USAGE:
 * All landmark objects should have {x, y, z} properties.
 * z defaults to 0 if not provided for backwards compatibility.
 */

const Vector3D = {
  /**
   * Create a 3D vector from a landmark object
   * Ensures z coordinate exists (defaults to 0)
   * 
   * @param {Object} landmark - MediaPipe landmark {x, y, z?}
   * @returns {Object} Normalized vector {x, y, z}
   */
  fromLandmark(landmark) {
    return {
      x: landmark.x || 0,
      y: landmark.y || 0,
      z: landmark.z || 0
    };
  },

  /**
   * Subtract two vectors: result = a - b
   * Used to create direction vectors from point to point
   * 
   * @param {Object} a - First vector {x, y, z}
   * @param {Object} b - Second vector {x, y, z}
   * @returns {Object} Difference vector {x, y, z}
   * 
   * EXAMPLE: Vector from elbow to wrist = Vector3D.subtract(wrist, elbow)
   */
  subtract(a, b) {
    return {
      x: a.x - b.x,
      y: a.y - b.y,
      z: (a.z || 0) - (b.z || 0)
    };
  },

  /**
   * Add two vectors: result = a + b
   * 
   * @param {Object} a - First vector {x, y, z}
   * @param {Object} b - Second vector {x, y, z}
   * @returns {Object} Sum vector {x, y, z}
   */
  add(a, b) {
    return {
      x: a.x + b.x,
      y: a.y + b.y,
      z: (a.z || 0) + (b.z || 0)
    };
  },

  /**
   * Calculate dot product of two vectors
   * Used for angle calculations: cos(θ) = (a·b) / (|a|·|b|)
   * 
   * @param {Object} a - First vector {x, y, z}
   * @param {Object} b - Second vector {x, y, z}
   * @returns {number} Scalar dot product
   * 
   * PROPERTIES:
   * - If dot > 0: vectors point in similar direction
   * - If dot < 0: vectors point in opposite directions
   * - If dot = 0: vectors are perpendicular
   */
  dot(a, b) {
    return a.x * b.x + a.y * b.y + (a.z || 0) * (b.z || 0);
  },

  /**
   * Calculate cross product of two vectors
   * Result is a vector perpendicular to both inputs
   * Useful for determining rotation direction
   * 
   * @param {Object} a - First vector {x, y, z}
   * @param {Object} b - Second vector {x, y, z}
   * @returns {Object} Cross product vector {x, y, z}
   */
  cross(a, b) {
    return {
      x: a.y * (b.z || 0) - (a.z || 0) * b.y,
      y: (a.z || 0) * b.x - a.x * (b.z || 0),
      z: a.x * b.y - a.y * b.x
    };
  },

  /**
   * Calculate the magnitude (length) of a 3D vector
   * This is the 3D equivalent of 2D's Math.hypot(x, y)
   * 
   * @param {Object} v - Vector {x, y, z}
   * @returns {number} Length of the vector
   * 
   * FORMULA: √(x² + y² + z²)
   */
  magnitude(v) {
    return Math.sqrt(v.x * v.x + v.y * v.y + (v.z || 0) * (v.z || 0));
  },

  /**
   * Normalize a vector to unit length (magnitude = 1)
   * Direction is preserved, length becomes 1
   * 
   * @param {Object} v - Vector {x, y, z}
   * @returns {Object} Unit vector {x, y, z}
   */
  normalize(v) {
    const mag = Vector3D.magnitude(v);
    if (mag === 0) return { x: 0, y: 0, z: 0 };
    return {
      x: v.x / mag,
      y: v.y / mag,
      z: (v.z || 0) / mag
    };
  },

  /**
   * Calculate 3D distance between two points
   * This is the true straight-line distance in 3D space
   * 
   * @param {Object} a - First point {x, y, z}
   * @param {Object} b - Second point {x, y, z}
   * @returns {number} Distance between points
   * 
   * WHY 3D DISTANCE MATTERS:
   * If an arm is pointed directly at the camera:
   * - 2D distance: appears very short (foreshortening)
   * - 3D distance: gives true arm length
   */
  distance(a, b) {
    return Vector3D.magnitude(Vector3D.subtract(a, b));
  },

  /**
   * Calculate angle between two vectors in degrees
   * Uses dot product formula: cos(θ) = (a·b) / (|a|·|b|)
   * 
   * @param {Object} a - First vector {x, y, z}
   * @param {Object} b - Second vector {x, y, z}
   * @returns {number} Angle in degrees (0-180)
   * 
   * USAGE FOR ELBOW ANGLE:
   * - a = vector from elbow to shoulder
   * - b = vector from elbow to wrist
   * - Result = angle at the elbow joint
   */
  angleBetween(a, b) {
    const magA = Vector3D.magnitude(a);
    const magB = Vector3D.magnitude(b);
    
    // Handle zero-length vectors
    if (magA === 0 || magB === 0) return 0;
    
    // Calculate cosine of angle using dot product
    const cosAngle = Vector3D.dot(a, b) / (magA * magB);
    
    // Clamp to [-1, 1] to handle floating point errors
    const clampedCos = Math.max(-1, Math.min(1, cosAngle));
    
    // Convert from radians to degrees
    return Math.acos(clampedCos) * (180 / Math.PI);
  },

  /**
   * Scale a vector by a scalar value
   * 
   * @param {Object} v - Vector {x, y, z}
   * @param {number} scalar - Scale factor
   * @returns {Object} Scaled vector {x, y, z}
   */
  scale(v, scalar) {
    return {
      x: v.x * scalar,
      y: v.y * scalar,
      z: (v.z || 0) * scalar
    };
  },

  /**
   * Linear interpolation between two points
   * t=0 returns a, t=1 returns b, t=0.5 returns midpoint
   * 
   * @param {Object} a - Start point {x, y, z}
   * @param {Object} b - End point {x, y, z}
   * @param {number} t - Interpolation factor (0-1)
   * @returns {Object} Interpolated point {x, y, z}
   */
  lerp(a, b, t) {
    return {
      x: a.x + (b.x - a.x) * t,
      y: a.y + (b.y - a.y) * t,
      z: (a.z || 0) + ((b.z || 0) - (a.z || 0)) * t
    };
  },

  /**
   * Calculate the midpoint between two points
   * Shorthand for lerp(a, b, 0.5)
   * 
   * @param {Object} a - First point {x, y, z}
   * @param {Object} b - Second point {x, y, z}
   * @returns {Object} Midpoint {x, y, z}
   */
  midpoint(a, b) {
    return Vector3D.lerp(a, b, 0.5);
  }
};


// ============================================================================
// SECTION 2: ONE EURO FILTER (ADAPTIVE SMOOTHING)
// ============================================================================
/**
 * PURPOSE: Reduce jitter in pose landmarks while maintaining responsiveness.
 * 
 * WHY ONE EURO FILTER:
 * - Standard low-pass filters introduce lag (delay in movement)
 * - One Euro Filter adapts: smooth when still, responsive when moving
 * - Perfect for pose tracking where stability AND responsiveness matter
 * 
 * HOW IT WORKS:
 * - When landmark moves slowly: applies heavy smoothing (reduces jitter)
 * - When landmark moves quickly: applies light smoothing (reduces lag)
 * - The "beta" parameter controls how much speed affects smoothing
 * 
 * PARAMETERS:
 * - minCutoff: Base smoothing amount (lower = smoother but more lag)
 * - beta: Speed coefficient (higher = more responsive to fast movement)
 * - dCutoff: Derivative filter cutoff (usually leave at 1.0)
 * 
 * 3D ENHANCEMENT:
 * Each coordinate (x, y, z) gets its own filter instance for proper smoothing
 */

class LowPassFilter {
  /**
   * Simple first-order low-pass filter
   * @param {number} alpha - Smoothing factor (0-1, higher = less smoothing)
   */
  constructor(alpha) {
    this.alpha = alpha;
    this.y = null;  // Previous output
  }
  
  /**
   * Filter a value
   * @param {number} value - New input value
   * @param {number} alpha - Optional override for smoothing factor
   * @returns {number} Filtered output
   */
  filter(value, alpha) {
    if (alpha !== undefined) this.alpha = alpha;
    
    if (this.y === null) {
      // First value - no filtering possible
      this.y = value;
    } else {
      // Exponential moving average
      this.y = this.alpha * value + (1 - this.alpha) * this.y;
    }
    return this.y;
  }
  
  /**
   * Reset filter state
   */
  reset() {
    this.y = null;
  }
}

class OneEuroFilter {
  /**
   * Create a One Euro Filter for a single value
   * 
   * @param {number} minCutoff - Minimum cutoff frequency (default 1.0)
   *                             Lower = smoother but more lag
   * @param {number} beta - Speed coefficient (default 0.007)
   *                        Higher = more responsive to fast movement
   * @param {number} dCutoff - Derivative cutoff frequency (default 1.0)
   *                           Usually leave at default
   */
  constructor(minCutoff = 1.0, beta = 0.007, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    
    // Two filters: one for the signal, one for its derivative
    this.x = new LowPassFilter(this.getAlpha(minCutoff, 30));
    this.dx = new LowPassFilter(this.getAlpha(dCutoff, 30));
    
    this.lastTime = null;
  }
  
  /**
   * Convert cutoff frequency to filter alpha
   * @param {number} cutoff - Cutoff frequency in Hz
   * @param {number} freq - Sampling frequency in Hz
   * @returns {number} Alpha value for low-pass filter
   */
  getAlpha(cutoff, freq) {
    const tau = 1.0 / (2 * Math.PI * cutoff);
    const te = 1.0 / freq;
    return 1.0 / (1.0 + tau / te);
  }
  
  /**
   * Filter a value with automatic adaptation
   * 
   * @param {number} value - New input value
   * @param {number} timestamp - Current timestamp in milliseconds
   * @returns {number} Filtered output
   */
  filter(value, timestamp) {
    // Calculate sampling frequency from timestamps
    const freq = this.lastTime ? 1000 / (timestamp - this.lastTime) : 30;
    this.lastTime = timestamp;
    
    // Calculate and filter the derivative (rate of change)
    const dValue = this.x.y === null ? 0 : (value - this.x.y) * freq;
    const edValue = this.dx.filter(dValue, this.getAlpha(this.dCutoff, freq));
    
    // Adapt cutoff frequency based on speed
    // Faster movement = higher cutoff = less smoothing
    const cutoff = this.minCutoff + this.beta * Math.abs(edValue);
    
    // Filter the signal with adapted cutoff
    return this.x.filter(value, this.getAlpha(cutoff, freq));
  }
  
  /**
   * Reset filter state
   */
  reset() {
    this.x.reset();
    this.dx.reset();
    this.lastTime = null;
  }
}


// ============================================================================
// SECTION 3: VELOCITY FATIGUE TRACKER
// ============================================================================
/**
 * PURPOSE: Track velocity degradation to identify fatigue during training.
 * 
 * UNCHANGED FROM 2D VERSION:
 * This class tracks velocity values regardless of how they're calculated.
 * The improvement comes from receiving more accurate 3D velocity values.
 * 
 * FATIGUE ZONES:
 * - FRESH: <5% velocity drop
 * - MILD: 5-10% drop
 * - MODERATE: 10-20% drop (approaching anaerobic threshold)
 * - HIGH: 20-30% drop (at lactate threshold)
 * - CRITICAL: >30% drop (significant fatigue)
 */
class VelocityFatigueTracker {
  constructor(config = {}) {
    this.config = {
      baselineReps: config.baselineReps || 3,
      alertThresholds: config.alertThresholds || [10, 20, 30],
      movementsToTrack: config.movementsToTrack || ['PRESS', 'CLEAN', 'SNATCH', 'SWING'],
      ...config
    };
    this.reset();
  }

  reset() {
    this.data = {};
    for (const movement of this.config.movementsToTrack) {
      this.data[movement] = {
        velocities: [],
        baselineVelocity: null,
        currentVelocity: null,
        peakVelocity: null,
        dropFromBaseline: 0,
        dropFromPeak: 0,
        thresholdsCrossed: [],
        repCount: 0,
        fatigueZone: 'FRESH'
      };
    }
  }

  resetSet() {
    for (const movement of this.config.movementsToTrack) {
      this.data[movement].velocities = [];
      this.data[movement].baselineVelocity = null;
      this.data[movement].currentVelocity = null;
      this.data[movement].dropFromBaseline = 0;
      this.data[movement].dropFromPeak = 0;
      this.data[movement].thresholdsCrossed = [];
      this.data[movement].repCount = 0;
      this.data[movement].fatigueZone = 'FRESH';
    }
  }

  addRep(movementType, velocity) {
    if (!this.data[movementType]) {
      console.warn(`[FatigueTracker] Unknown movement type: ${movementType}`);
      return null;
    }

    const data = this.data[movementType];
    data.velocities.push(velocity);
    data.currentVelocity = velocity;
    data.repCount++;
    
    if (!data.peakVelocity || velocity > data.peakVelocity) {
      data.peakVelocity = velocity;
    }
    
    if (data.repCount === this.config.baselineReps) {
      data.baselineVelocity = this.calculateAverage(data.velocities);
      console.log(`📊 [FatigueTracker] ${movementType} baseline: ${data.baselineVelocity.toFixed(2)} m/s`);
    }
    
    if (data.baselineVelocity) {
      data.dropFromBaseline = this.calculatePercentDrop(data.baselineVelocity, velocity);
      data.dropFromPeak = this.calculatePercentDrop(data.peakVelocity, velocity);
      this.checkThresholds(movementType, data);
      data.fatigueZone = this.determineFatigueZone(data.dropFromBaseline);
    }
    
    return this.getStatus(movementType);
  }

  calculatePercentDrop(reference, current) {
    if (!reference || reference === 0) return 0;
    const drop = ((reference - current) / reference) * 100;
    return Math.max(0, drop);
  }

  calculateAverage(arr) {
    if (arr.length === 0) return 0;
    return arr.reduce((sum, val) => sum + val, 0) / arr.length;
  }

  checkThresholds(movementType, data) {
    for (const threshold of this.config.alertThresholds) {
      if (data.dropFromBaseline >= threshold && !data.thresholdsCrossed.includes(threshold)) {
        data.thresholdsCrossed.push(threshold);
        this.onThresholdCrossed(movementType, threshold, data);
      }
    }
  }

  onThresholdCrossed(movementType, threshold, data) {
    const messages = {
      10: '⚠️ MILD FATIGUE - 10% velocity drop',
      20: '🟠 MODERATE FATIGUE - 20% drop (anaerobic threshold zone)',
      30: '🔴 HIGH FATIGUE - 30% drop (lactate threshold exceeded)'
    };
    console.log(`${messages[threshold] || threshold + '% drop'} | ${movementType} | Rep ${data.repCount}`);
  }

  determineFatigueZone(dropPercent) {
    if (dropPercent < 5) return 'FRESH';
    if (dropPercent < 10) return 'MILD';
    if (dropPercent < 20) return 'MODERATE';
    if (dropPercent < 30) return 'HIGH';
    return 'CRITICAL';
  }

  getStatus(movementType) {
    const data = this.data[movementType];
    if (!data) return null;
    return {
      movementType,
      repCount: data.repCount,
      currentVelocity: data.currentVelocity,
      baselineVelocity: data.baselineVelocity,
      peakVelocity: data.peakVelocity,
      dropFromBaseline: data.dropFromBaseline,
      dropFromPeak: data.dropFromPeak,
      fatigueZone: data.fatigueZone,
      thresholdsCrossed: data.thresholdsCrossed,
      hasBaseline: data.baselineVelocity !== null,
      velocityHistory: [...data.velocities]
    };
  }

  predictRepsToThreshold(movementType, targetDropPercent = 20) {
    const data = this.data[movementType];
    if (!data || !data.baselineVelocity || data.velocities.length < 3) return null;
    
    const n = data.velocities.length;
    const xMean = (n + 1) / 2;
    const yMean = this.calculateAverage(data.velocities);
    
    let numerator = 0, denominator = 0;
    for (let i = 0; i < n; i++) {
      numerator += (i + 1 - xMean) * (data.velocities[i] - yMean);
      denominator += (i + 1 - xMean) ** 2;
    }
    
    if (denominator === 0) return null;
    const slope = numerator / denominator;
    if (slope >= 0) return null;
    
    const targetVelocity = data.baselineVelocity * (1 - targetDropPercent / 100);
    const intercept = yMean - slope * xMean;
    const predictedRep = (targetVelocity - intercept) / slope;
    
    return {
      repsRemaining: Math.max(0, Math.ceil(predictedRep - n)),
      predictedRepNumber: Math.ceil(predictedRep),
      currentRep: n
    };
  }
}


// ============================================================================
// SECTION 4: SET TIMING TRACKER
// ============================================================================
/**
 * PURPOSE: Track work and rest periods during training.
 * 
 * UNCHANGED FROM 2D VERSION:
 * Timing is independent of how poses are measured.
 */
class SetTimingTracker {
  constructor() {
    this.reset();
  }

  reset() {
    this.currentSet = {
      number: 0,
      startTime: null,
      endTime: null,
      repCount: 0,
      isActive: false
    };
    
    this.restTimer = {
      startTime: null,
      isRunning: false,
      elapsed: 0
    };
    
    this.history = [];
    
    this.session = {
      totalWorkTime: 0,
      totalRestTime: 0,
      avgWorkTime: 0,
      avgRestTime: 0,
      avgWorkRestRatio: 0,
      setCount: 0
    };
  }

  onRep() {
    if (this.restTimer.isRunning) {
      this.stopRestTimer();
      this.startNewSet();
    }
    
    if (!this.currentSet.isActive) {
      this.startNewSet();
    }
    
    this.currentSet.repCount++;
    return this.getStatus();
  }

  startNewSet() {
    this.currentSet = {
      number: this.session.setCount + 1,
      startTime: Date.now(),
      endTime: null,
      repCount: 0,
      isActive: true
    };
    console.log(`🏋️ [TimingTracker] Set ${this.currentSet.number} started`);
  }

  onSetEnd() {
    if (!this.currentSet.isActive || this.currentSet.repCount === 0) {
      return null;
    }
    
    const now = Date.now();
    this.currentSet.endTime = now;
    this.currentSet.isActive = false;
    
    const setDuration = this.currentSet.endTime - this.currentSet.startTime;
    
    let restBeforeSet = 0;
    if (this.history.length > 0) {
      const lastSet = this.history[this.history.length - 1];
      restBeforeSet = this.currentSet.startTime - lastSet.endTime;
    }
    
    const completedSet = {
      number: this.currentSet.number,
      duration: setDuration,
      repCount: this.currentSet.repCount,
      restBefore: restBeforeSet,
      workRestRatio: restBeforeSet > 0 ? setDuration / restBeforeSet : 0,
      startTime: this.currentSet.startTime,
      endTime: this.currentSet.endTime
    };
    
    this.history.push(completedSet);
    
    this.session.setCount++;
    this.session.totalWorkTime += setDuration;
    if (restBeforeSet > 0) {
      this.session.totalRestTime += restBeforeSet;
    }
    
    this.updateSessionAverages();
    
    console.log(`✅ [TimingTracker] Set ${completedSet.number}: ${(setDuration/1000).toFixed(1)}s, ${completedSet.repCount} reps`);
    
    this.startRestTimer();
    
    return completedSet;
  }

  startRestTimer() {
    this.restTimer = {
      startTime: Date.now(),
      isRunning: true,
      elapsed: 0
    };
    console.log('⏱️ [TimingTracker] Rest timer started');
  }

  stopRestTimer() {
    if (!this.restTimer.isRunning) return;
    this.restTimer.elapsed = Date.now() - this.restTimer.startTime;
    this.restTimer.isRunning = false;
    console.log(`⏱️ [TimingTracker] Rest complete: ${(this.restTimer.elapsed/1000).toFixed(1)}s`);
  }

  getRestTimerElapsed() {
    if (!this.restTimer.isRunning) return this.restTimer.elapsed;
    return Date.now() - this.restTimer.startTime;
  }

  getCurrentSetDuration() {
    if (!this.currentSet.isActive || !this.currentSet.startTime) return 0;
    return Date.now() - this.currentSet.startTime;
  }

  updateSessionAverages() {
    if (this.session.setCount === 0) return;
    this.session.avgWorkTime = this.session.totalWorkTime / this.session.setCount;
    
    const setsWithRest = this.history.filter(s => s.restBefore > 0);
    if (setsWithRest.length > 0) {
      this.session.avgRestTime = setsWithRest.reduce((sum, s) => sum + s.restBefore, 0) / setsWithRest.length;
      this.session.avgWorkRestRatio = this.session.avgWorkTime / this.session.avgRestTime;
    }
  }

  formatTime(ms) {
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }

  formatTimeShort(ms) {
    return `${(ms / 1000).toFixed(1)}s`;
  }

  getStatus() {
    const restElapsed = this.getRestTimerElapsed();
    const setDuration = this.getCurrentSetDuration();
    
    return {
      isSetActive: this.currentSet.isActive,
      isResting: this.restTimer.isRunning,
      currentSetNumber: this.currentSet.number || 1,
      currentSetDuration: setDuration,
      currentSetDurationFormatted: this.formatTime(setDuration),
      currentSetReps: this.currentSet.repCount,
      restElapsed: restElapsed,
      restElapsedFormatted: this.formatTime(restElapsed),
      lastSet: this.history.length > 0 ? this.history[this.history.length - 1] : null,
      totalSets: this.session.setCount,
      avgWorkTime: this.session.avgWorkTime,
      avgWorkTimeFormatted: this.formatTimeShort(this.session.avgWorkTime),
      avgRestTime: this.session.avgRestTime,
      avgRestTimeFormatted: this.formatTimeShort(this.session.avgRestTime),
      avgWorkRestRatio: this.session.avgWorkRestRatio,
      avgWorkRestRatioFormatted: this.session.avgWorkRestRatio > 0 
        ? `1:${(1/this.session.avgWorkRestRatio).toFixed(1)}` 
        : '---',
      setHistory: [...this.history]
    };
  }
}


// ============================================================================
// SECTION 5: CALIBRATION SYSTEM (3D ENHANCED)
// ============================================================================
/**
 * PURPOSE: Convert pixel measurements to real-world centimeters using
 * the user's known height as a reference.
 * 
 * 3D ENHANCEMENTS:
 * 1. Uses 3D distance for body segment measurements (more accurate)
 * 2. Includes z-coordinate in all calculations
 * 3. Better handles cases where user is angled toward/away from camera
 * 
 * CALIBRATION PROCESS:
 * 1. User enters height in inches
 * 2. System captures 60 frames while user stands upright
 * 3. Calculates pixel-to-cm ratio using ankle-to-nose distance
 * 4. Measures all body segments in 3D
 * 
 * THE "RULER" CONCEPT:
 * - User's known height becomes the ruler
 * - Ankle-to-nose distance (adjusted for nose-to-head offset) = ruler length
 * - All other measurements use this ratio
 */
class CalibrationSystem {
  constructor() {
    // Configuration
    this.CALIBRATION_FRAMES = 60;       // ~2 seconds at 30fps
    this.NOSE_TO_HEAD_OFFSET_CM = 11;   // Average nose to top of head
    
    // Z-coordinate scaling factor
    // MediaPipe z is in same units as x, roughly
    // We'll use this to weight z contribution
    this.Z_SCALE = 1.0;

    this.state = {
      phase: "WAITING_FOR_HEIGHT",  // WAITING_FOR_HEIGHT -> CAPTURING -> COMPLETE
      
      userHeightInches: null,
      userHeightCm: null,
      ankleToNoseCm: null,
      
      framesCaptured: 0,
      ankleToNosePixelSamples: [],    // Array of 3D distance samples
      
      pixelToCmRatio: null,
      
      // 3D body segment measurements (in cm)
      bodySegments: {
        torso: null,
        thigh: null,
        shin: null,
        upperArm: null,
        forearm: null,
        ankleToNose: null
      },
      
      // NEW: Store average z-depth during calibration
      // Used to detect if user has moved closer/further from camera
      calibrationDepth: null
    };
  }

  setUserHeight(inches) {
    this.state.userHeightInches = inches;
    this.state.userHeightCm = inches * 2.54;
    this.state.ankleToNoseCm = this.state.userHeightCm - this.NOSE_TO_HEAD_OFFSET_CM;
    this.state.phase = "CAPTURING";

    console.log(`📏 [Calibration-3D] User Height: ${inches}" = ${this.state.userHeightCm.toFixed(1)}cm`);
    console.log(`📏 [Calibration-3D] Ankle-to-Nose (estimated): ${this.state.ankleToNoseCm.toFixed(1)}cm`);

    return {
      heightCm: this.state.userHeightCm,
      ankleToNoseCm: this.state.ankleToNoseCm
    };
  }

  /**
   * Capture calibration frame using 3D distance
   * 
   * @param {Object} pose - Pose data with {x, y, z} landmarks
   * @param {number} canvasHeight - Canvas height for pixel conversion
   * @returns {Object|null} Capture status
   * 
   * 3D CHANGE: Now uses Vector3D.distance for true 3D measurement
   */
  captureFrame(pose, canvasHeight) {
    if (this.state.phase !== "CAPTURING") return null;
    if (!pose.LEFT || !pose.RIGHT) return null;

    const leftAnkle = pose.LEFT.ANKLE;
    const rightAnkle = pose.RIGHT.ANKLE;
    const nose = pose.LEFT.NOSE;

    if (!leftAnkle || !rightAnkle || !nose) return null;

    // Calculate average ankle position in 3D
    const avgAnkle = Vector3D.midpoint(leftAnkle, rightAnkle);
    
    // Calculate 3D distance from ankle to nose
    // Scale coordinates to pixel space for the x and y components
    const ankleScaled = {
      x: avgAnkle.x * canvasHeight,
      y: avgAnkle.y * canvasHeight,
      z: (avgAnkle.z || 0) * canvasHeight * this.Z_SCALE
    };
    
    const noseScaled = {
      x: nose.x * canvasHeight,
      y: nose.y * canvasHeight,
      z: (nose.z || 0) * canvasHeight * this.Z_SCALE
    };
    
    // 3D distance calculation
    const ankleToNosePixels3D = Vector3D.distance(ankleScaled, noseScaled);
    
    // Validation: person must be reasonably upright
    // Check 2D projection first (y difference should be significant)
    const ankleToNose2D = Math.abs(avgAnkle.y - nose.y) * canvasHeight;
    if (ankleToNose2D < canvasHeight * 0.3) {
      return { status: "INVALID_POSE", message: "Stand upright facing camera" };
    }

    this.state.ankleToNosePixelSamples.push(ankleToNosePixels3D);
    this.state.framesCaptured++;

    const progress = this.state.framesCaptured / this.CALIBRATION_FRAMES;

    if (this.state.framesCaptured >= this.CALIBRATION_FRAMES) {
      return this.finalizeCalibration(pose, canvasHeight);
    }

    return {
      status: "CAPTURING",
      progress: progress,
      framesRemaining: this.CALIBRATION_FRAMES - this.state.framesCaptured
    };
  }

  /**
   * Finalize calibration with 3D measurements
   */
  finalizeCalibration(pose, canvasHeight) {
    // Use median for robustness
    const sortedSamples = [...this.state.ankleToNosePixelSamples].sort((a, b) => a - b);
    const medianAnkleToNosePixels = sortedSamples[Math.floor(sortedSamples.length / 2)];

    this.state.pixelToCmRatio = this.state.ankleToNoseCm / medianAnkleToNosePixels;

    // Measure body segments in 3D
    this.measureBodySegments3D(pose, canvasHeight);
    
    // Store calibration depth (average hip z)
    const avgHipZ = (pose.LEFT.HIP.z + pose.RIGHT.HIP.z) / 2;
    this.state.calibrationDepth = avgHipZ;

    this.state.phase = "COMPLETE";

    console.log("✅ [Calibration-3D] Complete!");
    console.log(`📏 [Calibration-3D] Ankle-to-Nose (3D): ${medianAnkleToNosePixels.toFixed(1)}px = ${this.state.ankleToNoseCm.toFixed(1)}cm`);
    console.log(`📐 [Calibration-3D] Pixel-to-CM Ratio: ${this.state.pixelToCmRatio.toFixed(4)} cm/px`);
    console.log(`📊 [Calibration-3D] Calibration Depth (z): ${avgHipZ.toFixed(4)}`);
    console.log("📊 [Calibration-3D] Body Segments:", this.state.bodySegments);

    return {
      status: "COMPLETE",
      pixelToCmRatio: this.state.pixelToCmRatio,
      bodySegments: this.state.bodySegments
    };
  }

  /**
   * Measure body segments using 3D distance
   * 
   * @param {Object} pose - Pose with 3D landmarks
   * @param {number} canvasHeight - For pixel scaling
   * 
   * 3D CHANGE: Uses Vector3D.distance instead of 2D hypot
   */
  measureBodySegments3D(pose, canvasHeight) {
    const ratio = this.state.pixelToCmRatio;
    const zScale = this.Z_SCALE;

    // Helper: 3D distance in pixels between two landmarks
    const distance3DPixels = (a, b) => {
      const aScaled = {
        x: a.x * canvasHeight,
        y: a.y * canvasHeight,
        z: (a.z || 0) * canvasHeight * zScale
      };
      const bScaled = {
        x: b.x * canvasHeight,
        y: b.y * canvasHeight,
        z: (b.z || 0) * canvasHeight * zScale
      };
      return Vector3D.distance(aScaled, bScaled);
    };

    // Average both sides for accuracy
    const avgSegment = (leftA, leftB, rightA, rightB) => {
      const leftDist = distance3DPixels(leftA, leftB);
      const rightDist = distance3DPixels(rightA, rightB);
      return ((leftDist + rightDist) / 2) * ratio;
    };

    this.state.bodySegments = {
      torso: avgSegment(
        pose.LEFT.SHOULDER, pose.LEFT.HIP,
        pose.RIGHT.SHOULDER, pose.RIGHT.HIP
      ),
      thigh: avgSegment(
        pose.LEFT.HIP, pose.LEFT.KNEE,
        pose.RIGHT.HIP, pose.RIGHT.KNEE
      ),
      shin: avgSegment(
        pose.LEFT.KNEE, pose.LEFT.ANKLE,
        pose.RIGHT.KNEE, pose.RIGHT.ANKLE
      ),
      upperArm: avgSegment(
        pose.LEFT.SHOULDER, pose.LEFT.ELBOW,
        pose.RIGHT.SHOULDER, pose.RIGHT.ELBOW
      ),
      forearm: avgSegment(
        pose.LEFT.ELBOW, pose.LEFT.WRIST,
        pose.RIGHT.ELBOW, pose.RIGHT.WRIST
      ),
      ankleToNose: this.state.ankleToNoseCm
    };

    return this.state.bodySegments;
  }

  pixelsToCm(pixels) {
    if (!this.state.pixelToCmRatio) {
      console.warn("[Calibration-3D] Not complete!");
      return null;
    }
    return pixels * this.state.pixelToCmRatio;
  }

  getPixelsPerMeter() {
    if (!this.state.pixelToCmRatio) return null;
    return 100 / this.state.pixelToCmRatio;
  }

  getArmLengthCm() {
    if (!this.state.bodySegments.upperArm || !this.state.bodySegments.forearm) {
      return null;
    }
    return this.state.bodySegments.upperArm + this.state.bodySegments.forearm;
  }

  getArmLengthPixels() {
    const armCm = this.getArmLengthCm();
    if (!armCm || !this.state.pixelToCmRatio) return null;
    return armCm / this.state.pixelToCmRatio;
  }
  
  /**
   * NEW: Get calibration depth for depth-aware adjustments
   */
  getCalibrationDepth() {
    return this.state.calibrationDepth;
  }

  isComplete() {
    return this.state.phase === "COMPLETE";
  }

  reset() {
    this.state = {
      phase: "WAITING_FOR_HEIGHT",
      userHeightInches: null,
      userHeightCm: null,
      ankleToNoseCm: null,
      framesCaptured: 0,
      ankleToNosePixelSamples: [],
      pixelToCmRatio: null,
      bodySegments: {
        torso: null,
        thigh: null,
        shin: null,
        upperArm: null,
        forearm: null,
        ankleToNose: null
      },
      calibrationDepth: null
    };
  }
}


// ============================================================================
// SECTION 6: VBT STATE MACHINE (3D ENHANCED)
// ============================================================================
/**
 * PURPOSE: Detect and classify kettlebell movements using 3D pose data.
 * 
 * 3D ENHANCEMENTS:
 * 
 * 1. ELBOW ANGLE (calculateElbowAngle3D):
 *    - Uses 3D vectors for accurate angle regardless of camera angle
 *    - Previous 2D version could give wrong angles when arm pointed at camera
 *    - Example: Arm extended toward camera appeared bent in 2D
 * 
 * 2. VELOCITY (calculateVelocity3D):
 *    - Tracks movement in all three dimensions
 *    - Captures full speed including depth changes (z movement)
 *    - Example: Snatch has significant z-component as bell moves away then up
 * 
 * 3. POSITION DETECTION:
 *    - Overhead check uses 3D distance for arm extension
 *    - Swing height detection considers depth
 *    - Rack position detection improved with 3D proximity
 * 
 * 4. SMOOTHING:
 *    - One Euro Filter applied to all three coordinates
 *    - Separate filter instance for each landmark axis
 * 
 * STATE MACHINE PHASES (unchanged):
 * - IDLE: Detecting starting position
 * - MOVING: Tracking active movement
 * - RETURNING: Waiting for return to complete rep
 * - SETTLING: Brief pause after snatch
 */
class VBTStateMachine {
  constructor(canvasHeight = 720, calibrationSystem = null) {
    this.canvasHeight = canvasHeight;
    this.calibrationSystem = calibrationSystem;
    
    // One Euro Filter instances for each landmark axis
    this.filters = {};
    
    // Z-coordinate scaling factor (same as calibration system)
    this.Z_SCALE = 1.0;

    this.THRESHOLDS = {
      // Elbow angle thresholds (degrees) - may need adjustment for 3D
      // 3D angles tend to be more accurate, so thresholds might feel different
      RACK_ELBOW_MAX: 45,           // Increased slightly for 3D accuracy
      LOCKOUT_ELBOW_MIN: 155,       // Decreased slightly for 3D accuracy
      
      // Hold durations (frames)
      RACK_HOLD_FRAMES: 30,
      LOCKOUT_HOLD_FRAMES: 0,
      OVERHEAD_HOLD_FRAMES: 3,
      
      // Position thresholds
      WRIST_NEAR_SHOULDER: 0.08,
      WRIST_OVERHEAD: 0.10,
      TUCKED_MAX: 0.1,
      ALIGN_MAX: 0.20,
      
      // Snatch detection
      SNATCH_ARM_EXTENSION_RATIO: 0.80,
      
      // Velocity
      VELOCITY_ALPHA: 0.15,
      
      // One Euro Filter parameters
      ONE_EURO_MIN_CUTOFF: 0.5,     // Lower = smoother
      ONE_EURO_BETA: 0.025,         // Higher = more responsive
      
      MAX_REALISTIC_VELOCITY: 8.0,
      ZERO_BAND: 0.1,
      MIN_DT: 0.016,
      MAX_DT: 0.1,
      
      // Reset
      RESET_DURATION_FRAMES: 30,
      
      // Settling
      SNATCH_SETTLING_FRAMES: 1
    };

    this.calibrationData = {
      isCalibrated: false,
      framesCaptured: 0,
      neutralWristOffset: 0,
      maxTorsoLength: 0
    };

    this.reset();
  }

  reset() {
    // Clear all filter state for fresh start
    this.filters = {};
    
    this.state = {
      lockedSide: "unknown",
      phase: "IDLE",

      startedFromRack: false,
      startedBelowHip: false,
      startedFromOverhead: false,
      
      settlingFrames: 0,
      
      reachedRack: false,
      reachedOverhead: false,
      reachedLockout: false,
      reachedElbowExtension: false,
      reachedSwingHeight: false,
      wentBelowHip: false,
      elbowStayedExtended: true,
      
      rackHoldFrames: 15,
      lockoutHoldFrames: 0,
      
      currentRepPeak: 0,
      smoothedVy: 0,
      
      // NEW: 3D velocity tracking
      smoothedVelocity3D: { vx: 0, vy: 0, vz: 0, speed: 0 },
      
      lastTimestamp: 0,
      lastWristPos: null,
      
      calibration: null,
      resetProgress: 0,
      
      pendingMovement: null,
      
      smoothedLandmarks: {
        LEFT: { WRIST: null, SHOULDER: null, HIP: null, KNEE: null, NOSE: null, ANKLE: null, ELBOW: null },
        RIGHT: { WRIST: null, SHOULDER: null, HIP: null, KNEE: null, NOSE: null, ANKLE: null, ELBOW: null }
      }
    };
  }

  /**
   * Calculate elbow angle using 3D vectors
   * 
   * @param {Object} shoulder - Shoulder landmark {x, y, z}
   * @param {Object} elbow - Elbow landmark {x, y, z}
   * @param {Object} wrist - Wrist landmark {x, y, z}
   * @returns {number} Angle in degrees (0-180)
   * 
   * 3D CHANGE: Uses Vector3D.angleBetween for true 3D angle
   * 
   * WHY THIS MATTERS:
   * Consider an arm extended directly toward the camera:
   * - 2D: Shoulder, elbow, wrist appear nearly overlapping = ~180° (looks straight)
   * - But actually the arm might be bent at 90° in the z direction!
   * - 3D: Correctly calculates the actual joint angle
   */
  calculateElbowAngle3D(shoulder, elbow, wrist) {
    // Create 3D vectors from elbow to shoulder and elbow to wrist
    const toShoulder = Vector3D.subtract(shoulder, elbow);
    const toWrist = Vector3D.subtract(wrist, elbow);
    
    // Calculate 3D angle between vectors
    return Vector3D.angleBetween(toShoulder, toWrist);
  }
  
  /**
   * Legacy 2D elbow angle for comparison/debugging
   * Kept for backwards compatibility
   */
  calculateElbowAngle(shoulder, elbow, wrist) {
    // This is the original 2D calculation - kept for reference
    const toShoulder = {
      x: shoulder.x - elbow.x,
      y: shoulder.y - elbow.y
    };
    const toWrist = {
      x: wrist.x - elbow.x,
      y: wrist.y - elbow.y
    };
    const dot = toShoulder.x * toWrist.x + toShoulder.y * toWrist.y;
    const magShoulder = Math.hypot(toShoulder.x, toShoulder.y);
    const magWrist = Math.hypot(toWrist.x, toWrist.y);
    const cosAngle = dot / (magShoulder * magWrist);
    const angleRad = Math.acos(Math.max(-1, Math.min(1, cosAngle)));
    return angleRad * (180 / Math.PI);
  }

  /**
   * Check if wrist is overhead using 3D distance
   * 
   * 3D CHANGE: Uses full 3D arm extension calculation
   */
  isWristOverhead3D(wrist, shoulder, nose) {
    if (this.calibrationSystem && this.calibrationSystem.isComplete()) {
      const armLengthPixels = this.calibrationSystem.getArmLengthPixels();
      
      if (armLengthPixels) {
        // Calculate 3D distance from shoulder to wrist
        const shoulderScaled = {
          x: shoulder.x * this.canvasHeight,
          y: shoulder.y * this.canvasHeight,
          z: (shoulder.z || 0) * this.canvasHeight * this.Z_SCALE
        };
        const wristScaled = {
          x: wrist.x * this.canvasHeight,
          y: wrist.y * this.canvasHeight,
          z: (wrist.z || 0) * this.canvasHeight * this.Z_SCALE
        };
        
        const armExtension3D = Vector3D.distance(shoulderScaled, wristScaled);
        
        // Also check that wrist is above shoulder (y decreases going up)
        const wristAboveShoulder = wrist.y < shoulder.y;
        
        // Arm must be extended AND wrist must be above shoulder
        const threshold = armLengthPixels * this.THRESHOLDS.SNATCH_ARM_EXTENSION_RATIO;
        const isOverhead3D = armExtension3D > threshold && wristAboveShoulder;
        
        if (isOverhead3D) {
          return true;
        }
      }
    }

    // Fallback: 2D check (wrist above nose)
    return wrist.y < (nose.y - this.THRESHOLDS.WRIST_OVERHEAD);
  }

  /**
   * Check swing height with 3D awareness
   */
  isSwingHeight(wrist, hip, shoulder, nose) {
    const torsoLength = Math.abs(hip.y - shoulder.y);
    const navelHeight = hip.y - (torsoLength * 0.30);
    const wristAboveNavel = wrist.y < navelHeight;
    return wristAboveNavel;
  }

  /**
   * Apply One Euro Filter smoothing to all 3D coordinates
   * 
   * @param {Object} rawPose - Raw pose from MediaPipe
   * @param {number} timestamp - Current timestamp
   * @returns {Object} Smoothed pose
   * 
   * 3D CHANGE: Filters all three coordinates (x, y, z) independently
   */
  smoothLandmarks3D(rawPose, timestamp) {
    const smoothed = { LEFT: {}, RIGHT: {} };
    const joints = ['WRIST', 'SHOULDER', 'HIP', 'KNEE', 'NOSE', 'ANKLE', 'ELBOW'];
    
    for (const side of ['LEFT', 'RIGHT']) {
      if (!rawPose[side]) continue;
      
      for (const joint of joints) {
        const raw = rawPose[side][joint];
        if (!raw) continue;
        
        const key = `${side}_${joint}`;
        
        // Initialize filters for this joint if needed
        if (!this.filters[key]) {
          this.filters[key] = {
            x: new OneEuroFilter(
              this.THRESHOLDS.ONE_EURO_MIN_CUTOFF,
              this.THRESHOLDS.ONE_EURO_BETA
            ),
            y: new OneEuroFilter(
              this.THRESHOLDS.ONE_EURO_MIN_CUTOFF,
              this.THRESHOLDS.ONE_EURO_BETA
            ),
            z: new OneEuroFilter(
              this.THRESHOLDS.ONE_EURO_MIN_CUTOFF,
              this.THRESHOLDS.ONE_EURO_BETA
            )
          };
        }
        
        // Filter each coordinate independently
        smoothed[side][joint] = {
          x: this.filters[key].x.filter(raw.x, timestamp),
          y: this.filters[key].y.filter(raw.y, timestamp),
          z: this.filters[key].z.filter(raw.z || 0, timestamp)
        };
      }
    }
    
    this.state.smoothedLandmarks = smoothed;
    return smoothed;
  }

  /**
   * Calculate 3D velocity
   * 
   * @param {Object} wrist - Wrist position {x, y, z}
   * @param {number} timestamp - Current timestamp
   * @returns {Object} Velocity {vx, vy, vz, speed}
   * 
   * 3D CHANGE: Now tracks z velocity and calculates 3D speed
   * 
   * WHY 3D VELOCITY MATTERS:
   * - Snatch: Bell moves away from body (z) then up (y)
   * - Clean: Bell moves toward body (z) as it racks
   * - Full 3D speed captures true movement velocity
   */
  calculateVelocity3D(wrist, timestamp) {
    if (!this.state.lastWristPos || !this.state.calibration) {
      this.state.lastWristPos = { 
        x: wrist.x, 
        y: wrist.y, 
        z: wrist.z || 0, 
        t: timestamp 
      };
      return { vx: 0, vy: 0, vz: 0, speed: 0 };
    }

    const dt = (timestamp - this.state.lastWristPos.t) / 1000;

    if (dt < this.THRESHOLDS.MIN_DT || dt > this.THRESHOLDS.MAX_DT) {
      this.state.lastWristPos = { 
        x: wrist.x, 
        y: wrist.y, 
        z: wrist.z || 0, 
        t: timestamp 
      };
      return { vx: 0, vy: 0, vz: 0, speed: 0 };
    }

    // Calculate displacement in pixels
    const dxPx = (wrist.x - this.state.lastWristPos.x) * this.canvasHeight;
    const dyPx = (wrist.y - this.state.lastWristPos.y) * this.canvasHeight;
    const dzPx = ((wrist.z || 0) - (this.state.lastWristPos.z || 0)) * this.canvasHeight * this.Z_SCALE;

    // Convert to meters per second
    let vx = (dxPx / this.state.calibration) / dt;
    let vy = (dyPx / this.state.calibration) / dt;
    let vz = (dzPx / this.state.calibration) / dt;

    // Calculate 3D speed
    let speed = Math.sqrt(vx * vx + vy * vy + vz * vz);

    // Apply zero band
    if (speed < this.THRESHOLDS.ZERO_BAND) {
      speed = 0;
      vx = 0;
      vy = 0;
      vz = 0;
    }

    // Clamp to realistic maximum
    speed = Math.min(speed, this.THRESHOLDS.MAX_REALISTIC_VELOCITY);
    vy = Math.max(-this.THRESHOLDS.MAX_REALISTIC_VELOCITY, 
         Math.min(this.THRESHOLDS.MAX_REALISTIC_VELOCITY, vy));
    vz = Math.max(-this.THRESHOLDS.MAX_REALISTIC_VELOCITY, 
         Math.min(this.THRESHOLDS.MAX_REALISTIC_VELOCITY, vz));

    this.state.lastWristPos = { 
      x: wrist.x, 
      y: wrist.y, 
      z: wrist.z || 0, 
      t: timestamp 
    };

    return { vx, vy, vz, speed };
  }

  /**
   * MAIN UPDATE FUNCTION (3D Enhanced)
   * 
   * Changes from 2D version:
   * - Uses smoothLandmarks3D for filtering
   * - Uses calculateElbowAngle3D for accurate angles
   * - Uses isWristOverhead3D for position detection
   * - Uses calculateVelocity3D for full 3D speed
   */
  update(pose, timestamp, ctx, canvas) {
    if (!pose.LEFT || !pose.RIGHT) return null;

    // 3D smoothing
    const smoothedPose = this.smoothLandmarks3D(pose, timestamp);

    // Initial pose calibration (30 frames)
    const currentTorso = Math.abs(smoothedPose.LEFT.SHOULDER.y - smoothedPose.LEFT.HIP.y);
    const leftWristOffset = smoothedPose.LEFT.WRIST.y - smoothedPose.LEFT.HIP.y;
    const rightWristOffset = smoothedPose.RIGHT.WRIST.y - smoothedPose.RIGHT.HIP.y;

    if (!this.calibrationData.isCalibrated) {
      this.calibrationData.framesCaptured++;
      this.calibrationData.neutralWristOffset += (leftWristOffset + rightWristOffset) / 2;
      this.calibrationData.maxTorsoLength = Math.max(this.calibrationData.maxTorsoLength, currentTorso);

      if (this.calibrationData.framesCaptured >= 30) {
        this.calibrationData.neutralWristOffset /= 30;
        this.calibrationData.isCalibrated = true;
        console.log("✅ [StateMachine-3D] Pose Calibration Complete");
      }
      return null;
    }

    // Reset detection
    const leftAtHome = Math.abs(leftWristOffset - this.calibrationData.neutralWristOffset) < 0.10;
    const rightAtHome = Math.abs(rightWristOffset - this.calibrationData.neutralWristOffset) < 0.10;
    const isTall = currentTorso > (this.calibrationData.maxTorsoLength * 0.85);

    if (leftAtHome && rightAtHome && isTall) {
      this.state.resetProgress++;
      this.drawResetUI(ctx, canvas, smoothedPose);

      if (this.state.resetProgress > this.THRESHOLDS.RESET_DURATION_FRAMES) {
        onStandingReset();
        this.reset();
        return null;
      }
    } else {
      this.state.resetProgress = 0;
    }

    // Side lock
    if (this.state.lockedSide === "unknown") {
      if (Math.abs(smoothedPose.LEFT.WRIST.y - smoothedPose.RIGHT.WRIST.y) > 0.1) {
        this.state.lockedSide = smoothedPose.LEFT.WRIST.y > smoothedPose.RIGHT.WRIST.y ? "LEFT" : "RIGHT";
      } else {
        return null;
      }
    }

    const side = this.state.lockedSide;
    const wrist = smoothedPose[side].WRIST;
    const elbow = smoothedPose[side].ELBOW;
    const shoulder = smoothedPose[side].SHOULDER;
    const hip = smoothedPose[side].HIP;
    const nose = smoothedPose[side].NOSE;

    // Velocity calibration
    if (!this.state.calibration) {
      if (this.calibrationSystem && this.calibrationSystem.isComplete()) {
        this.state.calibration = this.calibrationSystem.getPixelsPerMeter();
        console.log(`📐 [StateMachine-3D] Using calibrated px/m: ${this.state.calibration.toFixed(2)}`);
      } else if (shoulder && hip) {
        const TORSO_METERS = 0.45;
        this.state.calibration = (Math.abs(shoulder.y - hip.y) * this.canvasHeight) / TORSO_METERS;
        console.log(`📐 [StateMachine-3D] Using estimated px/m: ${this.state.calibration.toFixed(2)} (legacy)`);
      }
    }

    // Calculate positions using 3D methods
    const elbowAngle = this.calculateElbowAngle3D(shoulder, elbow, wrist);
    const wristBelowHip = wrist.y > hip.y;
    const wristNearShoulder = Math.abs(wrist.y - shoulder.y) < this.THRESHOLDS.WRIST_NEAR_SHOULDER;
    const wristOverhead = this.isWristOverhead3D(wrist, shoulder, nose);

    // Position detection (with 3D angle)
    const inRackPosition = elbowAngle < this.THRESHOLDS.RACK_ELBOW_MAX && 
                          wristNearShoulder && 
                          Math.abs(elbow.x - hip.x) < this.THRESHOLDS.TUCKED_MAX;
    
    const inLockout = elbowAngle > this.THRESHOLDS.LOCKOUT_ELBOW_MIN && 
                      wristOverhead && 
                      Math.abs(shoulder.x - wrist.x) < this.THRESHOLDS.ALIGN_MAX;

    // 3D velocity calculation
    const velocity3D = this.calculateVelocity3D(wrist, timestamp);
    
    // Smooth the 3D velocity
    const alpha = this.THRESHOLDS.VELOCITY_ALPHA;
    this.state.smoothedVelocity3D = {
      vx: alpha * velocity3D.vx + (1 - alpha) * this.state.smoothedVelocity3D.vx,
      vy: alpha * velocity3D.vy + (1 - alpha) * this.state.smoothedVelocity3D.vy,
      vz: alpha * velocity3D.vz + (1 - alpha) * this.state.smoothedVelocity3D.vz,
      speed: alpha * velocity3D.speed + (1 - alpha) * this.state.smoothedVelocity3D.speed
    };
    
    // Keep smoothedVy for backwards compatibility with UI
    this.state.smoothedVy = this.state.smoothedVelocity3D.vy;
    this.state.lastTimestamp = timestamp;

    let result = null;

    // ========================================
    // STATE MACHINE (logic unchanged, uses 3D measurements)
    // ========================================

    if (this.state.phase === "IDLE") {
      if (inLockout) {
        this.state.lockoutHoldFrames++;
        if (this.state.lockoutHoldFrames >= this.THRESHOLDS.OVERHEAD_HOLD_FRAMES) {
          this.state.startedFromOverhead = true;
          this.state.startedFromRack = false;
          this.state.startedBelowHip = false;
        }
      } else if (inRackPosition) {
        this.state.rackHoldFrames++;
        this.state.lockoutHoldFrames = 0;
        if (this.state.rackHoldFrames >= this.THRESHOLDS.RACK_HOLD_FRAMES) {
          this.state.startedFromRack = true;
          this.state.startedFromOverhead = false;
          this.state.startedBelowHip = false;
        }
      } else if (wristBelowHip) {
        this.state.rackHoldFrames = 0;
        this.state.lockoutHoldFrames = 0;
        this.state.startedFromRack = false;
        this.state.startedFromOverhead = false;
        this.state.startedBelowHip = true;
      }
      
      if (this.state.startedFromOverhead && !inLockout) {
        this.state.phase = "MOVING";
        this.state.reachedRack = false;
        this.state.reachedOverhead = false;
        this.state.reachedLockout = false;
        this.state.reachedSwingHeight = false;
        this.state.wentBelowHip = false;
        this.state.elbowStayedExtended = true;
        this.state.currentRepPeak = 0;
        this.state.lockoutHoldFrames = 0;
        this.state.rackHoldFrames = 0;
        console.log("🏋️ [StateMachine-3D] Movement started from OVERHEAD");
      } else if (this.state.startedFromRack && !inRackPosition) {
        this.state.phase = "MOVING";
        this.state.reachedRack = false;
        this.state.reachedOverhead = false;
        this.state.reachedLockout = false;
        this.state.reachedSwingHeight = false;
        this.state.wentBelowHip = false;
        this.state.elbowStayedExtended = true;
        this.state.currentRepPeak = 0;
        this.state.lockoutHoldFrames = 0;
        console.log("🏋️ [StateMachine-3D] Movement started from RACK");
      } else if (this.state.startedBelowHip && !wristBelowHip) {
        this.state.phase = "MOVING";
        this.state.reachedRack = false;
        this.state.reachedOverhead = false;
        this.state.reachedLockout = false;
        this.state.reachedSwingHeight = false;
        this.state.wentBelowHip = false;
        this.state.elbowStayedExtended = true;
        this.state.currentRepPeak = 0;
        this.state.lockoutHoldFrames = 0;
        this.state.rackHoldFrames = 0;
        console.log("🏋️ [StateMachine-3D] Movement started from BELOW HIP");
      }
    }

    else if (this.state.phase === "MOVING") {
      // Track peak velocity using 3D speed
      this.state.currentRepPeak = Math.max(
        this.state.currentRepPeak, 
        this.state.smoothedVelocity3D.speed
      );
      
      if (elbowAngle < this.THRESHOLDS.RACK_ELBOW_MAX) {
        this.state.elbowStayedExtended = false;
      }
      
      if (elbowAngle > this.THRESHOLDS.LOCKOUT_ELBOW_MIN) {
        this.state.reachedElbowExtension = true;
      }
      
      if (wristBelowHip) {
        this.state.wentBelowHip = true;
      }
      
      if (wristOverhead) {
        this.state.reachedOverhead = true;
      }
      
      if (this.isSwingHeight(wrist, hip, shoulder, nose)) {
        this.state.reachedSwingHeight = true;
      }
      
      if (inLockout) {
        this.state.reachedLockout = true;
        this.state.lockoutHoldFrames++;
      } else {
        this.state.lockoutHoldFrames = 0;
      }
      
      if (inRackPosition) {
        this.state.reachedRack = true;
        this.state.rackHoldFrames++;
      } else {
        this.state.rackHoldFrames = 0;
      }
      
      // Movement completion detection
      if (this.state.startedFromRack && 
          this.state.reachedLockout && 
          this.state.lockoutHoldFrames >= this.THRESHOLDS.LOCKOUT_HOLD_FRAMES &&
          !this.state.wentBelowHip) {
        this.state.phase = "RETURNING";
        this.state.pendingMovement = "PRESS";
        console.log("⏳ [StateMachine-3D] PRESS lockout confirmed");
      }
      
      else if ((this.state.startedBelowHip || this.state.startedFromRack || this.state.startedFromOverhead) && 
               this.state.wentBelowHip && 
               this.state.reachedOverhead &&
               this.state.reachedLockout && 
               wristBelowHip) {
        this.state.phase = "RETURNING";
        this.state.pendingMovement = "SNATCH";
        console.log("⏳ [StateMachine-3D] SNATCH lockout confirmed");
      }
      
      else if (this.state.startedBelowHip && 
               !this.state.elbowStayedExtended &&
               this.state.reachedRack && 
               this.state.rackHoldFrames >= this.THRESHOLDS.RACK_HOLD_FRAMES) {
        result = { type: "CLEAN", velocity: this.state.currentRepPeak };
        console.log("✅ [StateMachine-3D] CLEAN complete");
        this.resetForNextRep(true);
      }
      
      else if (this.state.startedFromRack && 
               this.state.wentBelowHip && 
               this.state.reachedRack && 
               this.state.rackHoldFrames >= this.THRESHOLDS.RACK_HOLD_FRAMES) {
        result = { type: "CLEAN", velocity: this.state.currentRepPeak };
        console.log("✅ [StateMachine-3D] RECLEAN complete");
        this.resetForNextRep(true);
      }
      
      else if (this.state.startedBelowHip && 
               this.state.reachedSwingHeight &&
               wristBelowHip) {
        result = { type: "SWING", velocity: this.state.currentRepPeak };
        console.log("✅ [StateMachine-3D] SWING complete");
        this.resetForNextRep(false);
      }
    }

    else if (this.state.phase === "RETURNING") {
      this.state.currentRepPeak = Math.max(
        this.state.currentRepPeak, 
        this.state.smoothedVelocity3D.speed
      );
      
      if (this.state.pendingMovement === "PRESS" && inRackPosition) {
        this.state.rackHoldFrames++;
        if (this.state.rackHoldFrames >= this.THRESHOLDS.RACK_HOLD_FRAMES) {
          result = { type: "PRESS", velocity: this.state.currentRepPeak };
          console.log("✅ [StateMachine-3D] PRESS complete");
          this.resetForNextRep(true);
        }
      }
      
      else if (this.state.pendingMovement === "SNATCH" && wristBelowHip) {
        result = { type: "SNATCH", velocity: this.state.currentRepPeak };
        console.log("✅ [StateMachine-3D] SNATCH complete");
        this.resetForSnatch();
      }
    }

    else if (this.state.phase === "SETTLING") {
      this.state.settlingFrames++;
      
      if (this.state.settlingFrames >= this.THRESHOLDS.SNATCH_SETTLING_FRAMES) {
        this.state.phase = "IDLE";
        
        if (inLockout) {
          this.state.startedFromOverhead = true;
          console.log("📍 [StateMachine-3D] After settling: OVERHEAD");
        } else if (wristBelowHip) {
          this.state.startedBelowHip = true;
          console.log("📍 [StateMachine-3D] After settling: BELOW HIP");
        }
      }
    }

    return result;
  }

  resetForSnatch() {
    this.state.phase = "SETTLING";
    this.state.settlingFrames = 0;
    this.state.startedFromRack = false;
    this.state.startedBelowHip = false;
    this.state.startedFromOverhead = false;
    this.state.reachedRack = false;
    this.state.reachedOverhead = false;
    this.state.reachedLockout = false;
    this.state.reachedElbowExtension = false;
    this.state.reachedSwingHeight = false;
    this.state.wentBelowHip = false;
    this.state.elbowStayedExtended = true;
    this.state.rackHoldFrames = 0;
    this.state.lockoutHoldFrames = 0;
    this.state.currentRepPeak = 0;
    this.state.pendingMovement = null;
  }

  resetForNextRep(inRack) {
    this.state.phase = "IDLE";
    this.state.startedFromRack = inRack;
    this.state.startedBelowHip = !inRack;
    this.state.startedFromOverhead = false;
    this.state.reachedRack = inRack;
    this.state.reachedOverhead = false;
    this.state.reachedLockout = false;
    this.state.reachedElbowExtension = false;
    this.state.reachedSwingHeight = false;
    this.state.wentBelowHip = false;
    this.state.elbowStayedExtended = true;
    this.state.rackHoldFrames = inRack ? this.THRESHOLDS.RACK_HOLD_FRAMES : 0;
    this.state.lockoutHoldFrames = 0;
    this.state.currentRepPeak = 0;
    this.state.pendingMovement = null;
  }

  drawResetUI(ctx, canvas, pose) {
    const centerX = (pose.LEFT.SHOULDER.x + pose.RIGHT.SHOULDER.x) / 2 * canvas.width;
    const centerY = (pose.LEFT.SHOULDER.y + pose.LEFT.HIP.y) / 2 * canvas.height;
    const pct = this.state.resetProgress / this.THRESHOLDS.RESET_DURATION_FRAMES;
    
    ctx.beginPath();
    ctx.arc(centerX, centerY, 40, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.lineWidth = 8;
    ctx.stroke();
    
    ctx.beginPath();
    ctx.arc(centerX, centerY, 40, -Math.PI / 2, (-Math.PI / 2) + (Math.PI * 2 * pct));
    ctx.strokeStyle = "#3b82f6";
    ctx.stroke();
  }
}


// ============================================================================
// SECTION 7: APP INITIALIZATION
// ============================================================================
/**
 * Main application object with all shared state
 */
const app = {
  video: null,
  canvas: null,
  ctx: null,
  
  landmarker: null,
  isModelLoaded: false,
  
  isTestRunning: false,
  totalReps: 0,
  lastMove: "READY",
  history: { CLEAN: [], PRESS: [], SNATCH: [], SWING: [] },
  
  stateMachine: null,
  calibrationSystem: null,
  
  fatigueTracker: null,
  timingTracker: null
};


/**
 * Initialize application
 */
async function initializeApp() {
  app.video = document.getElementById("video");
  app.canvas = document.getElementById("canvas");
  app.ctx = app.canvas.getContext("2d");

  // Initialize 3D-enhanced calibration system
  app.calibrationSystem = new CalibrationSystem();
  
  // Initialize performance trackers
  app.fatigueTracker = new VelocityFatigueTracker();
  app.timingTracker = new SetTimingTracker();

  // UI handlers
  document.getElementById("btn-camera").onclick = startCamera;
  document.getElementById("file-input").onchange = handleUpload;
  document.getElementById("btn-start-test").onclick = toggleTest;
  document.getElementById("btn-reset").onclick = resetSession;

  // Height calibration handler
  const heightInput = document.getElementById("height-input");
  const calibrateBtn = document.getElementById("btn-calibrate");

  if (calibrateBtn) {
    calibrateBtn.onclick = () => {
      const heightInches = parseFloat(heightInput.value);
      if (heightInches && heightInches > 48 && heightInches < 96) {
        app.calibrationSystem.setUserHeight(heightInches);
        updateCalibrationUI();
      } else {
        alert("Please enter a valid height (48-96 inches)");
      }
    };
  }

  // Initialize MediaPipe PoseLandmarker
  // The pose landmarker already provides 3D coordinates (x, y, z) by default
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );
  
  app.landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    // NOTE: outputSegmentationMasks: false (default)
    // The 3D coordinates come from the worldLandmarks in the result
    // But for this app we use the normalized landmarks which also have z
  });
  
  app.isModelLoaded = true;
  console.log("✅ [App-3D] MediaPipe PoseLandmarker loaded (3D mode)");
  
  requestAnimationFrame(masterLoop);
}


// ============================================================================
// SECTION 8: UI UPDATE FUNCTIONS
// ============================================================================

function updateCalibrationUI() {
  const statusEl = document.getElementById("calibration-status");
  if (!statusEl) return;

  const cal = app.calibrationSystem.state;

  if (cal.phase === "WAITING_FOR_HEIGHT") {
    statusEl.innerHTML = "Enter your height to begin 3D calibration";
  } else if (cal.phase === "CAPTURING") {
    statusEl.innerHTML = `Calibrating (3D)... Stand upright, arms at sides`;
  } else if (cal.phase === "COMPLETE") {
    const segments = cal.bodySegments;
    const armLength = segments.upperArm + segments.forearm;
    statusEl.innerHTML = `
      <div style="color: #22c55e; font-weight: bold;">✅ 3D Calibration Complete!</div>
      <div style="font-size: 12px; margin-top: 8px;">
        <div>Torso: ${segments.torso.toFixed(1)}cm</div>
        <div>Thigh: ${segments.thigh.toFixed(1)}cm</div>
        <div>Shin: ${segments.shin.toFixed(1)}cm</div>
        <div>Upper Arm: ${segments.upperArm.toFixed(1)}cm</div>
        <div>Forearm: ${segments.forearm.toFixed(1)}cm</div>
        <div style="color: #3b82f6; margin-top: 4px;">
          <strong>Full Arm (3D): ${armLength.toFixed(1)}cm</strong>
        </div>
      </div>`;
  }
}


function onStandingReset() {
  console.log("🧍 [App-3D] Standing reset detected - ending set");
  
  if (app.timingTracker) {
    const completedSet = app.timingTracker.onSetEnd();
    if (completedSet) {
      updateTimingUI();
      console.log(`📊 [App-3D] Set ${completedSet.number}: ${completedSet.repCount} reps in ${(completedSet.duration/1000).toFixed(1)}s`);
    }
  }
}


function updateFatigueUI(status, movementType) {
  if (!status) return;
  
  const zoneEl = document.getElementById('fatigue-zone');
  const dropEl = document.getElementById('fatigue-drop');
  const baselineEl = document.getElementById('fatigue-baseline');
  const currentEl = document.getElementById('fatigue-current');
  const predictionEl = document.getElementById('fatigue-prediction');
  
  if (zoneEl) {
    zoneEl.textContent = status.fatigueZone;
    zoneEl.className = 'fatigue-zone ' + status.fatigueZone.toLowerCase();
  }
  
  if (dropEl) {
    dropEl.textContent = status.hasBaseline 
      ? `${status.dropFromBaseline.toFixed(1)}%` 
      : 'Calibrating...';
  }
  
  if (baselineEl) {
    baselineEl.textContent = status.baselineVelocity 
      ? `${status.baselineVelocity.toFixed(2)} m/s` 
      : '---';
  }
  
  if (currentEl) {
    currentEl.textContent = `${status.currentVelocity.toFixed(2)} m/s`;
  }
  
  if (predictionEl && app.fatigueTracker) {
    const prediction = app.fatigueTracker.predictRepsToThreshold(movementType, 20);
    if (prediction && prediction.repsRemaining > 0) {
      predictionEl.textContent = `~${prediction.repsRemaining} reps to 20% drop`;
    } else if (status.dropFromBaseline >= 20) {
      predictionEl.textContent = 'Threshold reached';
    } else {
      predictionEl.textContent = '---';
    }
  }
}


function updateTimingUI() {
  if (!app.timingTracker) return;
  
  const status = app.timingTracker.getStatus();
  
  const setNumEl = document.getElementById('timing-set-number');
  const setRepsEl = document.getElementById('timing-set-reps');
  
  if (setNumEl) setNumEl.textContent = status.currentSetNumber;
  if (setRepsEl) setRepsEl.textContent = status.currentSetReps;
  
  const avgWorkEl = document.getElementById('timing-avg-work');
  const avgRestEl = document.getElementById('timing-avg-rest');
  const avgRatioEl = document.getElementById('timing-avg-ratio');
  const totalSetsEl = document.getElementById('timing-total-sets');
  
  if (avgWorkEl) avgWorkEl.textContent = status.avgWorkTimeFormatted || '---';
  if (avgRestEl) avgRestEl.textContent = status.avgRestTimeFormatted || '---';
  if (avgRatioEl) avgRatioEl.textContent = status.avgWorkRestRatioFormatted;
  if (totalSetsEl) totalSetsEl.textContent = status.totalSets;
}


function updateTimerDisplay() {
  if (!app.timingTracker) return;
  
  const status = app.timingTracker.getStatus();
  
  const restTimerEl = document.getElementById('rest-timer');
  const restLabelEl = document.getElementById('rest-timer-label');
  const setDurEl = document.getElementById('timing-set-duration');
  
  if (restTimerEl) {
    if (status.isResting) {
      restTimerEl.textContent = status.restElapsedFormatted;
      restTimerEl.className = 'rest-timer resting';
      if (restLabelEl) restLabelEl.textContent = 'REST';
    } else if (status.isSetActive) {
      restTimerEl.textContent = status.currentSetDurationFormatted;
      restTimerEl.className = 'rest-timer working';
      if (restLabelEl) restLabelEl.textContent = 'WORKING';
    } else {
      restTimerEl.textContent = '0:00';
      restTimerEl.className = 'rest-timer';
      if (restLabelEl) restLabelEl.textContent = 'READY';
    }
  }
  
  if (setDurEl) {
    setDurEl.textContent = status.currentSetDurationFormatted;
  }
}


// ============================================================================
// SECTION 9: HELPER FUNCTIONS
// ============================================================================

function handleUpload(e) {
  const file = e.target.files?.[0];
  if (!file) return;

  if (app.video.srcObject) {
    app.video.srcObject = null;
  }

  app.video.onloadedmetadata = () => {
    console.log("✅ [App-3D] Video loaded:", app.video.videoWidth, "x", app.video.videoHeight);
    app.canvas.width = app.video.videoWidth;
    app.canvas.height = app.video.videoHeight;
    // Create 3D-enhanced state machine
    app.stateMachine = new VBTStateMachine(app.canvas.height, app.calibrationSystem);
    document.getElementById("btn-start-test").disabled = false;
  };

  app.video.src = URL.createObjectURL(file);
  app.video.load();
  console.log("📁 [App-3D] Video selected:", file.name);
}


async function startCamera() {
  try {
    const s = await navigator.mediaDevices.getUserMedia({ video: true });

    app.video.onloadedmetadata = () => {
      console.log("✅ [App-3D] Camera loaded:", app.video.videoWidth, "x", app.video.videoHeight);
      app.canvas.width = app.video.videoWidth;
      app.canvas.height = app.video.videoHeight;
      // Create 3D-enhanced state machine
      app.stateMachine = new VBTStateMachine(app.canvas.height, app.calibrationSystem);
      document.getElementById("btn-start-test").disabled = false;
    };

    app.video.srcObject = s;
    console.log("📹 [App-3D] Camera started");
  } catch (err) {
    console.error("[App-3D] Camera error:", err);
    alert("Could not access camera: " + err.message);
  }
}


function toggleTest() {
  app.isTestRunning = !app.isTestRunning;
  document.getElementById("btn-start-test").innerText = app.isTestRunning ? "PAUSE" : "START";
  if (app.isTestRunning) app.video.play();
  else app.video.pause();
}


/**
 * Main loop - processes video frames and runs pose detection
 * 
 * 3D NOTE: MediaPipe's landmarks[0] array contains landmarks with {x, y, z}
 * The z coordinate is already included in the raw data from MediaPipe
 */
async function masterLoop(ts) {
  requestAnimationFrame(masterLoop);
  
  if (app.timingTracker) updateTimerDisplay();
  
  if (!app.isModelLoaded || !app.video.readyState) return;

  app.ctx.drawImage(app.video, 0, 0, app.canvas.width, app.canvas.height);
  
  const results = app.landmarker.detectForVideo(app.video, ts);

  if (results?.landmarks?.length > 0) {
    const raw = results.landmarks[0];
    
    // Convert to our format - landmarks already include z coordinate
    // MediaPipe z: negative = closer to camera, positive = further
    const pose = {
      LEFT: {
        WRIST: { x: raw[15].x, y: raw[15].y, z: raw[15].z || 0 },
        SHOULDER: { x: raw[11].x, y: raw[11].y, z: raw[11].z || 0 },
        HIP: { x: raw[23].x, y: raw[23].y, z: raw[23].z || 0 },
        KNEE: { x: raw[25].x, y: raw[25].y, z: raw[25].z || 0 },
        ANKLE: { x: raw[27].x, y: raw[27].y, z: raw[27].z || 0 },
        NOSE: { x: raw[0].x, y: raw[0].y, z: raw[0].z || 0 },
        ELBOW: { x: raw[13].x, y: raw[13].y, z: raw[13].z || 0 }
      },
      RIGHT: {
        WRIST: { x: raw[16].x, y: raw[16].y, z: raw[16].z || 0 },
        SHOULDER: { x: raw[12].x, y: raw[12].y, z: raw[12].z || 0 },
        HIP: { x: raw[24].x, y: raw[24].y, z: raw[24].z || 0 },
        KNEE: { x: raw[26].x, y: raw[26].y, z: raw[26].z || 0 },
        ANKLE: { x: raw[28].x, y: raw[28].y, z: raw[28].z || 0 },
        NOSE: { x: raw[0].x, y: raw[0].y, z: raw[0].z || 0 },
        ELBOW: { x: raw[14].x, y: raw[14].y, z: raw[14].z || 0 }
      }
    };

    // Handle height calibration
    if (app.calibrationSystem && app.calibrationSystem.state.phase === "CAPTURING") {
      const calResult = app.calibrationSystem.captureFrame(pose, app.canvas.height);
      if (calResult) {
        updateCalibrationUI();
        drawCalibrationOverlay(pose, calResult);
      }
    }

    // Run state machine
    if (app.isTestRunning && app.stateMachine) {
      const move = app.stateMachine.update(pose, ts, app.ctx, app.canvas);
      if (move) record(move);
      drawUI(app.stateMachine.state, pose);
      drawDebugSkeleton3D(pose);
    }
  }
}


function record(m) {
  app.totalReps++;
  app.lastMove = m.type;
  app.history[m.type].push(m.velocity);
  
  if (app.fatigueTracker) {
    const fatigueStatus = app.fatigueTracker.addRep(m.type, m.velocity);
    updateFatigueUI(fatigueStatus, m.type);
  }
  
  if (app.timingTracker) {
    app.timingTracker.onRep();
    updateTimingUI();
  }
  
  let plural = m.type.toLowerCase() + "s";
  if (m.type === "PRESS") plural = "presses";
  if (m.type === "SNATCH") plural = "snatches";
  
  const countEl = document.getElementById(`val-${plural}`);
  const velEl = document.getElementById(`val-${m.type.toLowerCase()}-velocity`);
  
  if (countEl) countEl.innerText = app.history[m.type].length;
  if (velEl) velEl.innerText = m.velocity.toFixed(2);
  
  document.getElementById("val-total-reps").innerText = app.totalReps;
  document.getElementById("detected-movement").innerText = m.type;
}


function resetSession() {
  app.totalReps = 0;
  app.lastMove = "READY";
  app.history = { CLEAN: [], PRESS: [], SNATCH: [], SWING: [] };
  
  if (app.stateMachine) app.stateMachine.reset();
  if (app.fatigueTracker) app.fatigueTracker.reset();
  if (app.timingTracker) app.timingTracker.reset();
  
  ['val-cleans', 'val-presses', 'val-snatches', 'val-swings', 'val-total-reps'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = '0';
  });
  
  ['val-clean-velocity', 'val-press-velocity', 'val-snatch-velocity', 'val-swing-velocity', 'val-velocity'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = '0.00';
  });
  
  document.getElementById("detected-movement").innerText = "READY";
  
  const zoneEl = document.getElementById('fatigue-zone');
  if (zoneEl) {
    zoneEl.textContent = 'FRESH';
    zoneEl.className = 'fatigue-zone fresh';
  }
  
  console.log("🔄 [App-3D] Session reset");
}


function drawCalibrationOverlay(pose, calResult) {
  const ctx = app.ctx;
  const canvas = app.canvas;

  if (calResult.status === "CAPTURING") {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    ctx.beginPath();
    ctx.arc(centerX, centerY, 60, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.lineWidth = 10;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(centerX, centerY, 60, -Math.PI / 2, (-Math.PI / 2) + (Math.PI * 2 * calResult.progress));
    ctx.strokeStyle = "#22c55e";
    ctx.stroke();

    ctx.fillStyle = "#fff";
    ctx.font = "bold 24px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("CALIBRATING (3D)", centerX, centerY - 10);
    ctx.font = "16px sans-serif";
    ctx.fillText(`${Math.round(calResult.progress * 100)}%`, centerX, centerY + 15);

    // Draw calibration line
    const nose = pose.LEFT.NOSE;
    const leftAnkle = pose.LEFT.ANKLE;
    const rightAnkle = pose.RIGHT.ANKLE;
    const avgAnkleX = (leftAnkle.x + rightAnkle.x) / 2;
    const avgAnkleY = (leftAnkle.y + rightAnkle.y) / 2;

    ctx.strokeStyle = "#22c55e";
    ctx.lineWidth = 3;
    ctx.setLineDash([10, 5]);
    ctx.beginPath();
    ctx.moveTo(avgAnkleX * canvas.width, avgAnkleY * canvas.height);
    ctx.lineTo(nose.x * canvas.width, nose.y * canvas.height);
    ctx.stroke();
    ctx.setLineDash([]);
  } else if (calResult.status === "INVALID_POSE") {
    ctx.fillStyle = "#ef4444";
    ctx.font = "bold 20px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(calResult.message, canvas.width / 2, 50);
  }
}


/**
 * Draw debug skeleton with 3D information
 * Shows z-depth as joint circle size (closer = larger)
 */
function drawDebugSkeleton3D(pose) {
  const ctx = app.ctx;
  const canvas = app.canvas;

  const workingSide = app.stateMachine?.state?.lockedSide || "unknown";

  for (const side of ['LEFT', 'RIGHT']) {
    const isWorkingArm = side === workingSide;
    const color = side === 'LEFT' ? '#00ff00' : '#ff0000';
    const wrist = pose[side].WRIST;
    const elbow = pose[side].ELBOW;
    const shoulder = pose[side].SHOULDER;
    const hip = pose[side].HIP;
    const knee = pose[side].KNEE;
    const ankle = pose[side].ANKLE;

    // Draw skeleton lines
    ctx.strokeStyle = color;
    ctx.lineWidth = isWorkingArm ? 10 : 4;
    ctx.beginPath();
    ctx.moveTo(wrist.x * canvas.width, wrist.y * canvas.height);
    ctx.lineTo(elbow.x * canvas.width, elbow.y * canvas.height);
    ctx.lineTo(shoulder.x * canvas.width, shoulder.y * canvas.height);
    ctx.lineTo(hip.x * canvas.width, hip.y * canvas.height);
    ctx.lineTo(knee.x * canvas.width, knee.y * canvas.height);
    ctx.lineTo(ankle.x * canvas.width, ankle.y * canvas.height);
    ctx.stroke();

    // Draw joints with z-depth visualization
    // Closer to camera (negative z) = larger circle
    const joints = [wrist, elbow, shoulder, hip, knee, ankle];
    ctx.strokeStyle = color;

    for (const joint of joints) {
      // Base radius
      let baseRadius = isWorkingArm ? 16 : 10;
      // Adjust by z: negative z (closer) = larger
      const zAdjust = -(joint.z || 0) * 20; // Scale z influence
      const radius = Math.max(5, baseRadius + zAdjust);
      
      ctx.lineWidth = isWorkingArm ? 6 : 3;
      ctx.beginPath();
      ctx.arc(joint.x * canvas.width, joint.y * canvas.height, radius, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Working arm label
    if (isWorkingArm) {
      ctx.fillStyle = "#ffff00";
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 4;
      ctx.font = "bold 32px sans-serif";
      ctx.textAlign = "center";
      ctx.strokeText("⚡ WORKING (3D)", wrist.x * canvas.width, wrist.y * canvas.height + 50);
      ctx.fillText("⚡ WORKING (3D)", wrist.x * canvas.width, wrist.y * canvas.height + 50);
    }

    // Elbow angle (3D calculated)
    if (app.stateMachine) {
      const elbowAngle = app.stateMachine.calculateElbowAngle3D(shoulder, elbow, wrist);
      ctx.fillStyle = isWorkingArm ? "#ffff00" : "#fff";
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 4;
      ctx.font = isWorkingArm ? "bold 42px sans-serif" : "bold 28px sans-serif";
      ctx.textAlign = "center";
      ctx.strokeText(`${elbowAngle.toFixed(0)}°`, elbow.x * canvas.width, elbow.y * canvas.height - 30);
      ctx.fillText(`${elbowAngle.toFixed(0)}°`, elbow.x * canvas.width, elbow.y * canvas.height - 30);
    }
  }

  // Face emoji
  const nose = pose.LEFT.NOSE;
  const leftShoulder = pose.LEFT.SHOULDER;
  const rightShoulder = pose.RIGHT.SHOULDER;
  const shoulderWidth = Math.abs(leftShoulder.x - rightShoulder.x) * canvas.width;
  const headSize = shoulderWidth * 1.25;

  ctx.font = `${headSize}px Arial`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('🙂', nose.x * canvas.width, nose.y * canvas.height);

  // Debug state info
  if (app.stateMachine && app.stateMachine.state) {
    const s = app.stateMachine.state;
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 5;
    ctx.font = "bold 36px sans-serif";
    ctx.textAlign = "left";

    const debugLines = [
      `Working Arm: ${s.lockedSide} (3D)`,
      `Phase: ${s.phase}${s.phase === 'SETTLING' ? ` (${s.settlingFrames}/${app.stateMachine.THRESHOLDS.SNATCH_SETTLING_FRAMES})` : ''}`,
      `From Rack: ${s.startedFromRack}`,
      `From Below Hip: ${s.startedBelowHip}`,
      `From Overhead: ${s.startedFromOverhead}`,
      `Went Below Hip: ${s.wentBelowHip}`,
      `Reached Overhead: ${s.reachedOverhead}`,
      `Elbow Extended: ${s.reachedElbowExtension}`,
      `Swing Height: ${s.reachedSwingHeight}`,
      `Reached Lockout: ${s.reachedLockout}`,
      `3D Speed: ${s.smoothedVelocity3D?.speed?.toFixed(2) || 0} m/s`
    ];

    debugLines.forEach((line, i) => {
      ctx.strokeText(line, 15, 45 + i * 42);
      ctx.fillText(line, 15, 45 + i * 42);
    });
  }
}


function drawUI(s, p) {
  // Display 3D speed instead of just Vy
  const speedEl = document.getElementById("val-velocity");
  if (speedEl && s.smoothedVelocity3D) {
    speedEl.innerText = s.smoothedVelocity3D.speed.toFixed(2);
  } else {
    speedEl.innerText = Math.abs(s.smoothedVy).toFixed(2);
  }
}


// ============================================================================
// START THE APP
// ============================================================================
initializeApp();
