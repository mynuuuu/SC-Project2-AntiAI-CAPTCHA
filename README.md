# SC-Project2-AntiAI-CAPTCHA

Anti AI CAPTCHA system built for Scalable Computing Main Project 2. A comprehensive slider-based CAPTCHA system with integrated behavior tracking and machine learning for human vs. bot detection.

## Features

### 🎯 Slider CAPTCHA System
- Interactive puzzle-based verification
- Multiple captcha instances
- Success/failure visual feedback
- Mobile and desktop support

### 📊 Behavior Tracking (NEW!)
- **Real-time mouse movement tracking**
- **Automatic CSV export** of behavior data
- **20+ metrics** captured per event (velocity, acceleration, direction, timing, etc.)
- **Visual recording indicator** during interaction
- Both human and bot behavior data collection

### 🤖 Machine Learning Models
- Random Forest classifier
- Gradient Boosting classifier
- Pre-trained models in `/models/`
- Training scripts in `/scripts/`

## Project Structure

```
SC-Project2-AntiAI-CAPTCHA/
├── src/                          # React application source
│   ├── App.js                    # Main app component
│   ├── CustomSliderCaptcha.js    # Slider captcha component with tracking
│   ├── useBehaviorTracking.js    # Behavior tracking hook (NEW!)
│   └── *.css                     # Styling files
├── behaviour_analysis/           # Standalone HTML behavior capture
│   ├── capture_behavior.html     # Original HTML tracker
│   ├── behavior_server.py        # Flask server for data collection
│   └── behavior_data/            # Collected behavior data
├── data/                         # Training datasets
│   ├── human_behavior_*.csv      # Human interaction data
│   └── bot_behavior.csv          # Bot interaction data
├── models/                       # Trained ML models
│   ├── rf_model.pkl              # Random Forest model
│   └── gb_model.pkl              # Gradient Boosting model
├── scripts/                      # Utility scripts
│   ├── train_model.py            # Model training script
│   ├── simulate_bots.py          # Bot behavior generator
│   └── ml_core.py                # ML utilities
└── BEHAVIOR_TRACKING.md          # Detailed tracking documentation (NEW!)
```

## Getting Started

### Prerequisites
- Node.js (v14 or higher)
- Python 3.9+ (for ML scripts and server)
- npm or yarn

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd SC-Project2-AntiAI-CAPTCHA
   ```

2. **Install React dependencies:**
   ```bash
   npm install
   ```

3. **Install Python dependencies (for ML and server):**
   ```bash
   cd behaviour_analysis
   pip install -r requirements.txt
   cd ..
   ```

### Running the Application

1. **Start the React app:**
   ```bash
   npm start
   ```
   The app will open at `http://localhost:3000`

2. **Optional: Start the behavior server (for server-side data collection):**
   ```bash
   cd behaviour_analysis
   python behavior_server.py
   ```
   Server runs at `http://localhost:5001`

## Using the Behavior Tracking

### Automatic CSV Export (Default)

When you solve any captcha:
1. Click and drag the slider button
2. A "Recording" indicator appears (red badge)
3. Your mouse movements are tracked
4. Upon completion (success or failure), a CSV file is **automatically downloaded**
5. Check your Downloads folder for files like:
   - `human_behavior_success_[SESSION_ID].csv`
   - `human_behavior_failed_[SESSION_ID].csv`

### Viewing Captured Statistics

After solving a captcha, the UI displays:
- Total events captured
- Duration of interaction
- Number of mouse moves
- Number of clicks
- CSV download confirmation

### What Data is Captured?

Each mouse event includes:
- **Position**: client, relative, page, and screen coordinates
- **Timing**: timestamps, time since start, time between events
- **Motion**: velocity (px/s), acceleration, direction (degrees)
- **Context**: event type, button states, modifier keys
- **Session**: unique ID, user agent, screen dimensions

See [BEHAVIOR_TRACKING.md](BEHAVIOR_TRACKING.md) for complete documentation.

## Machine Learning Pipeline

### 1. Collect Behavior Data
- Have humans solve captchas (data auto-downloads)
- Generate bot behavior using `/scripts/simulate_bots.py`
- Organize files in `/data/` folder

### 2. Train Models
```bash
cd scripts
python train_model.py
```

### 3. Evaluate Performance
Models are evaluated on:
- Accuracy
- Precision/Recall
- F1 Score
- ROC-AUC

### 4. Deploy
Integrate trained models (`.pkl` files) with your captcha system for real-time classification.

## Example Use Cases

### Research & Analysis
- Study human interaction patterns
- Compare bot vs human behavior
- Develop advanced detection algorithms

### Security
- Add extra verification layer
- Detect automated attacks
- Improve CAPTCHA resistance

### UX Optimization
- Analyze user friction points
- Optimize slider sensitivity
- Reduce false positives

## API Reference

### `useBehaviorTracking()` Hook

```javascript
const {
  isRecording,      // boolean: tracking active?
  sessionId,        // string: current session ID
  startRecording,   // function: start tracking
  stopRecording,    // function: stop tracking
  captureEvent,     // function: capture a single event
  saveToCSV,        // function: export to CSV file
  sendToServer,     // function: send to backend server
  getStats,         // function: get current statistics
  reset,            // function: reset tracking state
  events            // array: captured events
} = useBehaviorTracking();
```

### CSV Data Structure

See [BEHAVIOR_TRACKING.md](BEHAVIOR_TRACKING.md) for complete field documentation.

## Technologies Used

- **Frontend**: React.js, CSS3
- **Behavior Tracking**: Custom React Hooks, Canvas API
- **Backend**: Flask, Flask-CORS (optional)
- **Machine Learning**: scikit-learn, pandas, numpy
- **Data Processing**: Python, CSV

## Development

### Running Tests
```bash
npm test
```

### Building for Production
```bash
npm run build
```

### Linting
```bash
npm run lint
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Privacy & Ethics

This system is designed for research and legitimate security purposes. When deploying:
- Inform users about data collection
- Obtain necessary consents
- Follow data protection regulations (GDPR, CCPA, etc.)
- Implement appropriate data retention policies
- Anonymize sensitive information

## Troubleshooting

### CSV files not downloading?
- Check browser pop-up blocker settings
- Look in Downloads folder
- Check console for errors

### Recording not starting?
- Ensure you're dragging, not just clicking
- Check console logs for initialization
- Verify React hooks are working

### Server connection issues?
- Verify Flask server is running
- Check CORS configuration
- Confirm correct port (5001)

See [BEHAVIOR_TRACKING.md](BEHAVIOR_TRACKING.md) for more troubleshooting tips.

## License

[Add your license here]

## Authors

- Scalable Computing Project 2 Team

## Acknowledgments

- Thanks to all contributors
- Inspired by modern CAPTCHA systems
- Built for educational and research purposes 