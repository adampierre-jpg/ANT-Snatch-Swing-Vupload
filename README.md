# 🏋️ Kettlebell Velocity Tracker

A real-time velocity tracking web application for detecting anaerobic threshold via velocity drop-off during kettlebell exercises. Uses MediaPipe Pose for accurate wrist tracking and velocity calculations.

## Features

- **Real-time Velocity Tracking**: Track kettlebell movement velocity using AI-powered pose detection
- **Dual Input Modes**:
  - Live camera tracking for real-time workouts
  - Video upload support for analyzing recorded sessions (30fps 1080p, up to 5+ minutes)
- **Anaerobic Threshold Detection**: Automatic alerts at 20% and 25% velocity drop-off from baseline
- **Multiple Exercise Types**:
  - Snatch (Left/Right)
  - Two-Arm Swing
  - Swing (Left/Right)
- **Automatic Rep Counting**: Intelligent rep detection based on velocity patterns
- **Visual Feedback**:
  - Real-time skeleton overlay on video
  - Color-coded status indicators (green/yellow/red)
  - Live velocity metrics display
- **Results Tracking**: Save and review test history in browser localStorage
- **Mobile Responsive**: Works on iOS Safari and Chrome Android
- **No Backend Required**: Pure client-side processing for privacy and speed

## Technical Stack

- **Vanilla JavaScript** - No frameworks, optimized for performance
- **MediaPipe Pose Landmarker** - AI-powered pose detection (LITE model)
- **HTML5 Canvas** - Real-time visual feedback overlay
- **CSS3** - Responsive design with dark theme
- **LocalStorage API** - Client-side data persistence

## How It Works

### Velocity Calculation

1. **Pose Detection**: MediaPipe tracks 33 body landmarks at 30 FPS
2. **Wrist Tracking**: Monitors right wrist (landmark 16), left wrist (15), or both for two-arm exercises
3. **Pixel Calibration**: Automatically calibrates pixel-to-meter ratio using shoulder width (~0.4m)
4. **Velocity Formula**:
   ```
   distance = √((x2-x1)² + (y2-y1)²) × scaleFactor
   velocity = distance / timeDelta
   ```
5. **Smoothing**: 3-frame rolling average to reduce noise

### Anaerobic Threshold Detection

1. **Baseline Calculation**: Average of first 2-3 rep peak velocities
2. **Drop-off Monitoring**: Tracks percentage decrease from baseline
3. **Alert Thresholds**:
   - **15-20%**: Yellow warning (approaching threshold)
   - **20%**: Orange alert (threshold approaching)
   - **25%**: Red critical alert (stop test recommended)

### Rep Detection Logic

- **Rep Start**: Velocity crosses 0.3 m/s threshold (ascending)
- **Peak Tracking**: Monitors highest velocity during rep
- **Rep Complete**: Velocity drops below 0.15 m/s (50% of threshold)

## Usage Instructions

### Getting Started

1. **Select Exercise Type**
   - Choose your kettlebell exercise (Snatch Left/Right, Two-Arm Swing, etc.)

2. **Choose Input Mode**
   - **Live Camera**: Use device camera for real-time tracking
   - **Upload Video**: Analyze pre-recorded video (supports MP4, MOV, WebM)

3. **Grant Camera Permission** (Live mode only)
   - Browser will request camera access
   - Allow permission for front-facing camera

4. **Wait for Initialization**
   - MediaPipe model loads (~2-3 seconds)
   - "Start Test" button enables when ready

5. **Start Your Test**
   - Click "Start Test"
   - Perform kettlebell exercises in view of camera
   - Watch real-time metrics update

6. **Monitor Metrics**
   - **Current Velocity**: Instantaneous wrist speed
   - **Peak Velocity**: Highest velocity recorded
   - **Baseline Velocity**: Average of first 2-3 rep peaks
   - **Velocity Drop**: Percentage decrease from baseline
   - **Rep Count**: Automatically detected reps
   - **Status**: Color-coded indicator (green/yellow/red)

7. **Stop and Review**
   - Click "Stop Test" when finished
   - Results automatically saved to browser storage
   - Review past tests in Results section

### Best Practices

#### Camera Setup
- Position camera 6-10 feet away
- Ensure full body is visible (head to knees minimum)
- Good lighting conditions (front-lit, avoid backlighting)
- Stable camera position (tripod or phone stand)

#### During Test
- Warm up with 2-3 reps to establish baseline
- Maintain consistent form throughout
- Continue until 20-25% velocity drop
- Stop test if form degrades or fatigue is excessive

#### Video Upload Tips
- Use 30 FPS or higher
- Minimum 720p resolution (1080p recommended)
- MP4 format for best compatibility
- Keep file size under 100MB for smooth playback
- Ensure entire kettlebell movement is in frame

### Interpreting Results

#### Status Indicators
- **Green (Optimal)**: < 15% drop - maintaining velocity well
- **Yellow (Warning)**: 15-20% drop - approaching threshold
- **Red (Critical)**: > 20% drop - anaerobic threshold reached

#### Velocity Drop Percentage
- **0-10%**: Normal fatigue, continue
- **10-15%**: Moderate fatigue, monitor closely
- **15-20%**: Significant fatigue, approaching limit
- **20-25%**: Anaerobic threshold reached
- **> 25%**: Stop test, recovery needed

## Deployment to Vercel

### Method 1: GitHub Integration (Recommended)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit: KB Velocity Tracker"
   git push origin main
   ```

2. **Connect to Vercel**
   - Visit [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Vercel auto-detects static site (no configuration needed)

3. **Deploy**
   - Click "Deploy"
   - Wait 30-60 seconds
   - Your app is live!

### Method 2: Vercel CLI

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Deploy**
   ```bash
   vercel
   ```

3. **Follow Prompts**
   - Link to existing project or create new
   - Accept default settings
   - Deployment complete!

### Method 3: Drag & Drop

1. Visit [vercel.com/new](https://vercel.com/new)
2. Drag project folder to upload area
3. Deploy automatically

### Configuration

The `vercel.json` file includes required headers:
- `Cross-Origin-Embedder-Policy`: Required for MediaPipe SharedArrayBuffer
- `Cross-Origin-Opener-Policy`: Security isolation
- `Access-Control-Allow-Origin`: CORS support

No build commands or environment variables needed - it's a static site!

## Browser Compatibility

### Supported Browsers
- ✅ Chrome/Edge 90+ (Desktop & Mobile)
- ✅ Safari 14+ (Desktop & iOS)
- ✅ Firefox 88+
- ✅ Chrome Android 90+

### Required Features
- MediaDevices API (camera access)
- Canvas 2D Context
- LocalStorage
- ES6+ JavaScript
- WebAssembly (for MediaPipe)

### Mobile Support
- **iOS**: Safari 14+, Chrome iOS
- **Android**: Chrome 90+, Samsung Internet
- Front-facing camera required
- Minimum 720p camera resolution

## File Structure

```
kb-velocity-tracker/
├── index.html          # Main HTML structure
├── style.css           # Responsive styles & themes
├── app.js              # Core application logic
├── vercel.json         # Vercel deployment config
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Key Implementation Details

### MediaPipe Configuration
```javascript
{
  baseOptions: {
    modelAssetPath: "pose_landmarker_lite.task",
    delegate: "GPU"  // Hardware acceleration
  },
  runningMode: "VIDEO",
  numPoses: 1,
  minPoseDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
}
```

### Velocity Smoothing
- 3-frame rolling average
- Reduces sensor noise and jitter
- Maintains responsiveness

### Calibration
- Automatic shoulder width detection
- Reference: 0.4m average shoulder width
- Dynamically scales pixel measurements to meters

### Rep Detection
- State machine: `isInRep` flag
- Threshold: 0.3 m/s minimum velocity
- Hysteresis: 50% threshold for exit

## Troubleshooting

### "No Pose Detected" Warning
- **Solution**: Ensure full body is visible in frame
- Step back or widen camera angle
- Improve lighting conditions

### Camera Permission Denied
- **Solution**: Check browser settings
- Reset site permissions
- Use HTTPS (required for camera access)

### MediaPipe Fails to Load
- **Solution**: Check internet connection
- Verify CORS headers in `vercel.json`
- Try different browser
- Clear browser cache

### Velocity Appears Too High/Low
- **Solution**:
  - Ensure shoulders are visible for calibration
  - Stand 6-10 feet from camera
  - Check camera is level (not angled)

### Video Upload Not Working
- **Solution**:
  - Use MP4 format
  - Check file size (< 100MB recommended)
  - Ensure 30 FPS video
  - Try different browser

### Results Not Saving
- **Solution**:
  - Check localStorage not disabled
  - Clear browser cache/cookies
  - Disable private/incognito mode

## Performance Optimization

- Uses MediaPipe LITE model (3x faster than FULL)
- GPU acceleration enabled
- Throttles UI updates to 30 FPS
- Canvas-only rendering (no DOM manipulation in loop)
- LocalStorage for zero-latency data access

## Privacy & Data

- **100% Client-Side**: No data sent to servers
- **No Tracking**: No analytics or cookies
- **Local Storage Only**: Results stored in browser
- **No Account Required**: Use immediately
- **Camera Access**: Only when you grant permission
- **Offline Ready**: Works without internet (after initial load)

## Future Enhancements

Potential features for future versions:
- Export results to CSV/PDF
- Advanced analytics and charts
- Training program suggestions
- Multi-user profiles
- Cloud sync (optional)
- Additional exercise types
- Real-time coaching feedback

## Credits

Built with:
- [MediaPipe](https://developers.google.com/mediapipe) - Google's ML framework
- [Vercel](https://vercel.com) - Hosting platform

## License

MIT License - Free to use and modify

## Support

For issues or questions:
1. Check browser console for errors
2. Verify camera/video requirements met
3. Test on different browser
4. Check GitHub Issues (if open source)

---

**Built for athletes, coaches, and fitness enthusiasts to optimize kettlebell training through velocity-based threshold detection.**
