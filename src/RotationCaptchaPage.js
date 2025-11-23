import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import AnimalRotationCaptcha from './AnimalRotationCaptcha';
import './App.css';

function RotationCaptchaPage() {
  const navigate = useNavigate();
  const [isCompleted, setIsCompleted] = useState(false);

  const handleBack = () => {
    navigate('/');
  };

  const handleNext = () => {
    navigate('/animal-rotation-2');
  };

  const handleSuccess = () => {
    setIsCompleted(true);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Turing Tester</h1>
        <p>Complete the rotation captcha</p>
      </header>

      <div style={{ maxWidth: '600px', margin: '0 auto', padding: '20px' }}>
        <button
          onClick={handleBack}
          style={{
            marginBottom: '20px',
            padding: '10px 24px',
            fontSize: '16px',
            fontWeight: '600',
            color: '#667eea',
            background: 'white',
            border: '2px solid #667eea',
            borderRadius: '8px',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
          onMouseOver={(e) => {
            e.target.style.background = '#667eea';
            e.target.style.color = 'white';
          }}
          onMouseOut={(e) => {
            e.target.style.background = 'white';
            e.target.style.color = '#667eea';
          }}
        >
          ← Back
        </button>
        <AnimalRotationCaptcha onSuccess={handleSuccess} />

        {isCompleted && (
          <div style={{ marginTop: '30px', textAlign: 'center' }}>
            <button
              onClick={handleNext}
              style={{
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
    </div>
  );
}

export default RotationCaptchaPage;
