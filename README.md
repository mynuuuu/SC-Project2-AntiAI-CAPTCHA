# SC-Project2-AntiAI-CAPTCHA

Anti AI CAPTCHA system built for Scalable Computing Main Project 2. A comprehensive slider-based CAPTCHA system with integrated behavior tracking and machine learning for human vs. bot detection.

### Prerequisites
- Node.js (v14 or higher)
- Python 3.10+ (Python 3.9 has compatibility issues with google-generativeai)
- npm or yarn

### Quick Start - Run Demo/Emulation

To run the complete demo with frontend, backend, and all attackers:

```bash
./runme.sh
```

This script will:
1. Install all npm and Python dependencies
2. Start the backend server (Flask on port 5001)
3. Start the frontend (React on port 3000)
4. Run all attackers sequentially:
   - Computer Vision attacker
   - LLM Sycophancy attacker (requires `GEMINI_API_KEY` environment variable)
   - Universal LLM attacker (requires `GEMINI_API_KEY` environment variable)

**Note**: For LLM attackers to run, set the `GEMINI_API_KEY` environment variable:
```bash
export GEMINI_API_KEY="your_api_key_here" && ./runme.sh
```

The script will keep the frontend and backend running after all attackers complete. Press Ctrl+C to stop.

### Manual Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd SC-Project2-AntiAI-CAPTCHA
   ```

2. **Install React dependencies:**
   ```bash
   npm install
   ```

3. **Install Python dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Running the Application Manually

**IMPORTANT**: You must start the server first for behavior tracking to work.

1. **Start the Flask server (Required):**
   ```bash
   cd behaviour_analysis
   python behavior_server.py
   ```
   By default the server binds to `http://localhost:5001`. You can override the port,
   data directory, and CSV filename with environment variables (see *Environment Variables*).
   
   Keep this terminal open!

2. **Start the React app (in a new terminal):**
   ```bash
   # from project root
   export REACT_APP_API_BASE_URL=http://localhost:5001
   npm start
   ```
   The React app reads `REACT_APP_API_BASE_URL` to know where to send behavior data.
   When deploying to Vercel, set this env var to your public backend URL.
   The app will open at `http://localhost:3000`

See [SERVER_SETUP.md](SERVER_SETUP.md) for detailed instructions and
[`RENDER_DEPLOYMENT.md`](RENDER_DEPLOYMENT.md) for Render hosting steps.

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `REACT_APP_API_BASE_URL` | `http://localhost:5001` | React/Vercel frontend base URL for the Flask API |
| `PORT` | `5001` | Port for `behavior_server.py` (Render sets this automatically) |
| `FLASK_DEBUG` | `false` | Enables Flask debug mode locally |
| `DATA_DIR` | `<project_root>/data` | Directory where CSV/session files are written |
| `CSV_FILENAME` | `user_behavior_events.csv` | Only used by the legacy `/save_events` endpoint; captcha-specific data still writes to `data/captcha1.csv`, `captcha2.csv`, `captcha3.csv`, etc. |
| `SERVICE_NAME` | `Behavior Data Collection Server` | Printed on server start |
| `BEHAVIOR_SERVER_URL` | `http://localhost:5001/save_captcha_events` | Used by attacker tooling; override when pointing to prod |

Create `.env.local` (React) and `.env` (backend) files as needed to override these values for development vs production.

## Using the Behavior Tracking

### Server-Based Data Collection

The system saves behavior data to three specific CSV files in the `data/` folder:

**Files:**
- `data/captcha1.csv` - All attempts for Captcha 1
- `data/captcha2.csv` - All attempts for Captcha 2
- `data/captcha3.csv` - All attempts for Captcha 3

**Workflow:**
1. Click and drag the slider button
2. A "Recording" indicator appears (red badge)
3. Your mouse movements are tracked
4. Upon completion, data is sent to server
5. Data is **appended** to the appropriate CSV file
6. Captcha **auto-resets** (ready for next attempt)

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

## Technologies Used

- **Frontend**: React.js, CSS3
- **Behavior Tracking**: Custom React Hooks, Canvas API
- **Backend**: Flask, Flask-CORS (optional)
- **Machine Learning**: scikit-learn, pandas, numpy
- **Data Processing**: Python, CSV

## Privacy & Ethics

This system is designed for research and legitimate security purposes. When deploying:
- Inform users about data collection
- Obtain necessary consents
- Follow data protection regulations (GDPR, CCPA, etc.)
- Implement appropriate data retention policies
- Anonymize sensitive information

## Acknowledgments

- Thanks to all contributors
- Inspired by modern CAPTCHA systems
- Built for educational and research purposes
