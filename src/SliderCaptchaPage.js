import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import CustomSliderCaptcha from './CustomSliderCaptcha';
import './App.css';

function SliderCaptchaPage() {
  const navigate = useNavigate();
  const [captchaStatus, setCaptchaStatus] = useState({
    captcha1: false,
    captcha2: false,
    captcha3: false,
  });

  const [resetKeys, setResetKeys] = useState({
    captcha1: 0,
    captcha2: 0,
    captcha3: 0,
  });

  // Pool of available images
  const imagePool = useMemo(() => [
    'https://picsum.photos/seed/mountain1/400/200',
    'https://picsum.photos/seed/nature2/400/200',
    'https://picsum.photos/seed/landscape3/400/200',
    'https://picsum.photos/seed/beach4/400/200',
    'https://picsum.photos/seed/forest5/400/200',
    'https://picsum.photos/seed/city6/400/200',
    'https://picsum.photos/seed/sunset7/400/200',
    'https://picsum.photos/seed/ocean8/400/200',
    'https://picsum.photos/seed/desert9/400/200',
    'https://picsum.photos/seed/lake10/400/200',
  ], []);

  // Randomly select 3 images from the pool (only once on component mount)
  const captchaImages = useMemo(() => {
    // Shuffle the image pool
    const shuffled = [...imagePool].sort(() => Math.random() - 0.5);

    // Select the first 3 images
    const selectedImages = shuffled.slice(0, 3);

    return [
      {
        id: 'captcha1',
        title: 'Captcha 1',
        bgUrl: selectedImages[0],
      },
      {
        id: 'captcha2',
        title: 'Captcha 2',
        bgUrl: selectedImages[1],
      },
      {
        id: 'captcha3',
        title: 'Captcha 3',
        bgUrl: selectedImages[2],
      }
    ];
  }, [imagePool]);

  const handleVerify = (captchaId, data) => {
    console.log(`${captchaId} verification data:`, data);

    if (data.success) {
      setCaptchaStatus(prev => ({ ...prev, [captchaId]: true }));
    }
  };

  const resetCaptcha = (captchaId) => {
    // Only reset if not verified
    if (!captchaStatus[captchaId]) {
      setCaptchaStatus(prev => ({ ...prev, [captchaId]: false }));
      setResetKeys(prev => ({ ...prev, [captchaId]: prev[captchaId] + 1 }));
    }
  };

  const allCaptchasVerified = Object.values(captchaStatus).every(status => status === true);

  const handleNext = () => {
    navigate('/rotation-captcha');
  };

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
                onVerify={(data) => handleVerify(captcha.id, data)}
                onReset={() => resetCaptcha(captcha.id)}
              />
            </div>
            {captchaStatus[captcha.id] && (
              <div className="captcha-status">
                <div className="status-verified">
                  <span>✓ Verified Successfully!</span>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {allCaptchasVerified && (
        <div className="success-message">
          <h2>🎉 All Slider Captchas Verified!</h2>
          <p>Click Next to continue to the rotation captcha.</p>
          <button
            onClick={handleNext}
            style={{
              marginTop: '20px',
              padding: '12px 32px',
              fontSize: '18px',
              fontWeight: '600',
              color: 'white',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              border: 'none',
              borderRadius: '25px',
              cursor: 'pointer',
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
              transition: 'all 0.2s ease',
            }}
            onMouseOver={(e) => e.target.style.transform = 'translateY(-2px)'}
            onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
}

export default SliderCaptchaPage;
