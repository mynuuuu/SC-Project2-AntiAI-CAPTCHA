import React from 'react';
import { useNavigate } from 'react-router-dom';
import AnimalRotationCaptcha from './AnimalRotationCaptcha';
import './App.css';

function RotationCaptchaPage() {
  const navigate = useNavigate();

  const handleBack = () => {
    navigate('/');
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
        <AnimalRotationCaptcha />
      </div>
    </div>
  );
}

export default RotationCaptchaPage;
