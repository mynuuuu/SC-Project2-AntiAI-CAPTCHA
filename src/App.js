import { useState } from 'react';
import './App.css';
import CustomSliderCaptcha from './CustomSliderCaptcha';

function App() {
  const [captchaStatus, setCaptchaStatus] = useState({
    captcha1: false,
    captcha2: false,
    captcha3: false
  });

  const [resetKeys, setResetKeys] = useState({
    captcha1: 0,
    captcha2: 0,
    captcha3: 0
  });

  const [behaviorStats, setBehaviorStats] = useState({
    captcha1: null,
    captcha2: null,
    captcha3: null
  });

  // Different background images for each captcha
  const captchaImages = [
    {
      id: 'captcha1',
      title: 'Captcha 1',
      bgUrl: 'https://picsum.photos/seed/mountain1/400/200',
    },
    {
      id: 'captcha2',
      title: 'Captcha 2',
      bgUrl: 'https://picsum.photos/seed/nature2/400/200',
    },
    {
      id: 'captcha3',
      title: 'Captcha 3',
      bgUrl: 'https://picsum.photos/seed/landscape3/400/200',
    }
  ];

  const handleVerify = (captchaId, data) => {
    console.log(`${captchaId} verification data:`, data);

    if (data.success) {
      setCaptchaStatus(prev => ({ ...prev, [captchaId]: true }));
      setBehaviorStats(prev => ({ 
        ...prev, 
        [captchaId]: data.behaviorStats 
      }));
    } else {
      console.log(`${captchaId} failed. Behavior stats:`, data.behaviorStats);
    }
  };

  const resetCaptcha = (captchaId) => {
    setCaptchaStatus(prev => ({ ...prev, [captchaId]: false }));
    setResetKeys(prev => ({ ...prev, [captchaId]: prev[captchaId] + 1 }));
  };

  const allCaptchasVerified = Object.values(captchaStatus).every(status => status === true);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Turing Tester</h1>
        <p>Complete all captchas to verify you're human</p>
      </header>

      <div className="captcha-container">
        {captchaImages.map((captcha) => (
          <div key={captcha.id} className="captcha-item">
            <h3>{captcha.title}</h3>
            <div className="captcha-wrapper">
              <CustomSliderCaptcha
                key={resetKeys[captcha.id]}
                imageUrl={captcha.bgUrl}
                captchaId={captcha.id}
                onVerify={(data) => handleVerify(captcha.id, data)}
                onReset={() => resetCaptcha(captcha.id)}
              />
            </div>
            {captchaStatus[captcha.id] && (
              <div className="captcha-status">
                <div className="status-verified">
                  <span>✓ Verified Successfully!</span>
                  {behaviorStats[captcha.id] && (
                    <div style={{ 
                      marginTop: '10px', 
                      fontSize: '12px', 
                      color: '#666',
                      background: '#f8f9fa',
                      padding: '10px',
                      borderRadius: '5px'
                    }}>
                      <strong>Behavior Data Captured:</strong>
                      <div>Events: {behaviorStats[captcha.id].eventCount}</div>
                      <div>Duration: {behaviorStats[captcha.id].duration}s</div>
                      <div>Mouse Moves: {behaviorStats[captcha.id].moves}</div>
                      <div>Clicks: {behaviorStats[captcha.id].clicks}</div>
                      <div style={{ marginTop: '5px', fontStyle: 'italic', color: '#28a745' }}>
                        💾 Data saved to {captcha.id}.csv
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {allCaptchasVerified && (
        <div className="success-message">
          <h2>🎉 All Captchas Verified!</h2>
          <p>You have successfully completed all verifications.</p>
        </div>
      )}
    </div>
  );
}

export default App;
