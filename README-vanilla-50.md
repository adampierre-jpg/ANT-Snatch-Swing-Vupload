# VBT Vanilla 50 - Simple Movement Tracker

## Overview
A simplified version of the VBT (Velocity-Based Training) system focused on basic rep counting and velocity tracking. This "vanilla" version removes complex movement classification and focuses on core functionality.

## Features

### Core Tracking
✅ **Simple Rep Detection** - Basic up/down movement tracking  
✅ **Velocity Measurement** - Real-time velocity in m/s  
✅ **Peak Velocity** - Session peak tracking  
✅ **Rep History** - List of all reps with timestamps  

### Simplified Design
- No complex movement classification (swings, cleans, snatches, presses)
- No glycolytic fatigue tracking
- No multiple hand tracking
- Single-side tracking (right hand/wrist)
- Basic rep counter based on velocity thresholds

## Files

- `vbt-vanilla-50.html` - Simple user interface
- `vbt-vanilla-50.js` - Core tracking engine
- `vbt-vanilla-50-style.css` - Clean styling

## Usage

### Basic Setup
1. Open `vbt-vanilla-50.html` in a web browser
2. Upload a video or start your camera
3. Click "Start Tracking"
4. Perform movements
5. Export data as JSON

### Rep Detection Logic
Reps are detected based on:
- **Start**: Downward velocity > 0.5 m/s
- **Peak**: Maximum velocity during upward movement
- **End**: Velocity drops below 0.3 m/s after moving at least 0.2m upward

## Configuration

All settings can be adjusted in the `CONFIG` object in `vbt-vanilla-50.js`:

```javascript
const CONFIG = {
  REP_START_THRESHOLD: 0.5,     // m/s - moving down
  REP_END_THRESHOLD: 0.3,       // m/s - slowing down at top
  MIN_REP_HEIGHT: 0.2,          // meters above starting position
  SMOOTHING_ALPHA: 0.2,         // Velocity smoothing
};
```

## Export Format

```json
{
  "sessionDate": "2025-12-23T12:00:00.000Z",
  "totalReps": 10,
  "peakVelocity": 2.5,
  "reps": [
    {
      "number": 1,
      "velocity": 2.3,
      "timestamp": "2025-12-23T12:00:05.000Z"
    }
  ]
}
```

## Differences from Full VBT v3.5

### Removed Features
- ❌ Multiple movement type detection (swing/clean/snatch/press)
- ❌ Glycolytic fatigue tracking
- ❌ Dual-hand tracking
- ❌ Re-clean detection
- ❌ Zone-based position detection
- ❌ Make.com webhook integration
- ❌ Complex state machine

### Simplified Features
- ✅ Single rep counter (no movement classification)
- ✅ Basic velocity tracking
- ✅ Simple JSON export
- ✅ Right-hand only tracking
- ✅ Minimal configuration

## Use Cases

### Perfect For
- Basic velocity tracking
- Simple rep counting
- Learning VBT concepts
- Quick testing
- Minimal overhead applications

### Not Suitable For
- Complex movement classification
- Professional coaching programs
- Multi-athlete tracking
- Advanced fatigue analysis

## Technical Notes

### Performance
- Runs at 30 FPS target
- GPU-accelerated MediaPipe
- Lightweight compared to full version

### Privacy
- All processing client-side
- No video uploaded
- Local JSON export only

## Troubleshooting

**Reps Not Counting**
- Ensure adequate movement range (>0.2m)
- Check velocity exceeds thresholds
- Verify good camera angle (side view)

**Velocity Seems Wrong**
- Ensure proper lighting
- Check wrist landmark detection
- Verify full body visible in frame

## Version Info

**Version**: Vanilla 50  
**Base**: VBT v3.5  
**Date**: December 23, 2025  
**Purpose**: Simplified movement tracking
