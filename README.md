# VBT v3.5 - Complete Movement Detection System

## Overview
Advanced velocity-based training system with comprehensive kettlebell movement detection for Essential Fitness coaching programs.

## Features

### Movement Detection
✅ **Swing** - Ballistic hip hinge to shoulder height  
✅ **Clean from Floor** - Floor to rack position (first clean)  
✅ **Re-Clean** - Rack to backswing to rack (continuous work)  
✅ **Press** - Rack/backswing to overhead (slow, controlled)  
✅ **Snatch** - Floor to overhead (explosive)  

### Rep Tracking
- Separate counters for cleans and presses
- Distinguishes clean from floor vs re-clean
- Tracks continuous work patterns (Tempered style)
- Independent velocity baselines for each movement type

### Glycolytic Fatigue Tracking
- Real-time velocity drop-off percentage
- Separate baselines for cleans and presses
- Color-coded warnings (green/yellow/red)
- Perfect for monitoring Tempered program fatigue

### Zone-Based Detection
**FLOOR** - Below knee (starting position)  
**BACKSWING** - Below hip, above knee (transition)  
**RACK** - At shoulder height, close to torso  
**OVERHEAD** - Above shoulder (lockout)  

## Files

### Core System
- `vbt-v3.5-COMPLETE.js` - Main detection engine
- `vbt-v3.5.html` - User interface
- `vbt-v3.5-styles.css` - Styling

### Configuration
All tunable thresholds in `CONFIG` object:

```javascript
MOVEMENT: {
  PRESS_VELOCITY_THRESHOLD: 1.5,  // Below = press, Above = ballistic
  SNATCH_MIN_HEIGHT_ABOVE_SHOULDER: 0.05,
  CLEAN_HORIZONTAL_PROXIMITY: 0.18,
  // ... etc
}
```

## Usage

### Basic Setup
1. Open `vbt-v3.5.html` in browser
2. Upload video or start camera
3. Click "Start Test"
4. Perform movements
5. Export data to Make.com

### Movement Examples

**Tempered Program (1 Clean + Multiple Presses)**
```
1. Floor → Rack (CLEAN FROM FLOOR)
2. Rack → Overhead (PRESS #1)
3. Overhead → Rack
4. Rack → Backswing → Rack (RE-CLEAN)
5. Rack → Overhead (PRESS #2)
```

**Continuous Cleans**
```
1. Floor → Rack (CLEAN FROM FLOOR)
2. Rack → Backswing → Rack (RE-CLEAN #1)
3. Rack → Backswing → Rack (RE-CLEAN #2)
```

**Snatch Test**
```
1. Floor → Overhead (SNATCH)
2. Overhead → Floor
3. Floor → Overhead (SNATCH)
```

## Detection Logic

### Clean Classification
**From Floor**: Starts below knee → Lockout at rack  
**Re-Clean**: Starts in backswing (from rack drop) → Lockout at rack

### Press vs Snatch
**Press**: Peak velocity < 1.5 m/s (controlled grind)  
**Snatch**: Peak velocity ≥ 1.5 m/s (explosive ballistic)

### State Machine Flow
```
IDLE/LOCKOUT
  ↓
FLOOR or BACKSWING (detect starting position)
  ↓
BOTTOM (waiting for upward pull)
  ↓
CONCENTRIC (tracking velocity and position)
  ↓
LOCKOUT (2-frame confirmation at rack or overhead)
  ↓
Classify movement based on:
  - Starting position (floor vs rack)
  - Ending position (rack vs overhead)
  - Peak velocity (slow vs fast)
```

## Export Structure

```json
{
  "athlete_id": "dad_ready_user",
  "session_date": "2025-12-18T...",
  "sets": [
    {
      "set_order": 1,
      "hand": "right",
      "cleans": [
        {"type": "CLEAN_FROM_FLOOR", "velocity": 2.8},
        {"type": "RE_CLEAN", "velocity": 2.7}
      ],
      "presses": [
        {"type": "PRESS", "velocity": 1.2},
        {"type": "PRESS", "velocity": 1.15}
      ],
      "summary": {
        "total_cleans": 2,
        "floor_cleans": 1,
        "re_cleans": 1,
        "total_presses": 2
      }
    }
  ]
}
```

## Tuning Guide

### Press Detection Too Sensitive
**Problem**: Slow snatches classified as presses  
**Fix**: Increase `PRESS_VELOCITY_THRESHOLD` (1.5 → 1.8)

### Re-Cleans Not Detected
**Problem**: Backswing zone too narrow  
**Fix**: Check knee landmark visibility in video

### Rack Position False Positives
**Problem**: Swings classified as cleans  
**Fix**: Adjust `CLEAN_HORIZONTAL_PROXIMITY` (0.18 → 0.15)

## Business Applications

### Dad Ready Assessment
- Track snatch test velocity
- Identify movement asymmetries
- Baseline functional capacity

### Tempered Programming
- Monitor press velocity decline (glycolytic stress)
- Track clean capacity separately
- Optimize rest intervals based on drop-off

### Progress Tracking
- Session-to-session velocity improvements
- Movement-specific work capacity
- Fatigue resistance development

## Technical Notes

### Performance
- Runs at 30 FPS target
- GPU-accelerated MediaPipe
- Real-time physics calculations

### Accuracy
- 2-frame lockout confirmation
- Zone-based position verification
- Velocity-smoothed classification

### Privacy
- All processing client-side
- No video uploaded
- Only metrics exported

## Troubleshooting

**Movements Not Counting**
- Check camera angle (side view best)
- Ensure knee landmarks visible
- Verify adequate lighting

**Wrong Movement Classification**
- Enable DEBUG_MODE in CONFIG
- Check console logs for zone/velocity values
- Adjust thresholds based on your movement patterns

**Premature Set Ending**
- Increase RESET_GRACE_MS_AFTER_LOCK (default 5000ms)
- Ensure standing motion is deliberate

## Next Steps

### Immediate (Client Testing)
- Obfuscate code (protect IP)
- Test with real client videos
- Document threshold adjustments

### Short Term (Backend Migration)
- Move logic to Python backend
- Create coach dashboard
- Implement user authentication

### Long Term (Platform)
- Mobile app (React Native)
- Automated programming recommendations
- Multi-client analytics

---

**Version**: 3.5  
**Date**: December 18, 2025  
**Status**: Production Ready  
**Use Case**: Essential Fitness kettlebell coaching programs
