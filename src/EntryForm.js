import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

export default function EntryForm() {
  const navigate = useNavigate();
  const location = useLocation();
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [nextEnabled, setNextEnabled] = useState(false);

  // Check if user passed CAPTCHA (from localStorage set by MorseCaptcha)
  useEffect(() => {
    const captchaPassed = localStorage.getItem('captcha_passed') === 'true';
    const isHuman = localStorage.getItem('is_human') === 'true';
    
    if (captchaPassed && isHuman) {
      setNextEnabled(true);
    }
  }, []);

  const handleVerifyCaptcha = () => {
    // Clear any previous CAPTCHA results
    localStorage.removeItem('captcha_passed');
    localStorage.removeItem('is_human');
    localStorage.removeItem('final_classification');
    localStorage.removeItem('slider_classification');
    setNextEnabled(false);
    
    // Navigate to CAPTCHA system (starts with slider)
    navigate('/slider-captcha');
  };

  const handleNext = () => {
    if (nextEnabled) {
      // Clear CAPTCHA flags
      localStorage.removeItem('captcha_passed');
      localStorage.removeItem('is_human');
      
      // Navigate to homepage
      navigate('/home');
    }
  };

  return (
    <div style={{ 
      maxWidth: '500px', 
      margin: '100px auto', 
      padding: '40px',
      border: '2px solid #667eea',
      borderRadius: '12px',
      boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
    }}>
      <h1 style={{ textAlign: 'center', marginBottom: '30px', color: '#667eea' }}>
        Login Form
      </h1>
      
      <div style={{ marginBottom: '20px' }}>
        <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600' }}>
          Email:
        </label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          style={{
            width: '100%',
            padding: '12px',
            fontSize: '16px',
            border: '2px solid #ddd',
            borderRadius: '8px',
            boxSizing: 'border-box'
          }}
          placeholder="Enter your name"
        />
      </div>

      <div style={{ marginBottom: '30px' }}>
        <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600' }}>
          Password:
        </label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          style={{
            width: '100%',
            padding: '12px',
            fontSize: '16px',
            border: '2px solid #ddd',
            borderRadius: '8px',
            boxSizing: 'border-box'
          }}
          placeholder="Enter your password"
        />
      </div>

      <div style={{ display: 'flex', gap: '15px', marginBottom: '15px' }}>
        <button
          onClick={handleVerifyCaptcha}
          style={{
            flex: 1,
            padding: '14px',
            fontSize: '16px',
            fontWeight: '600',
            color: 'white',
            background: '#667eea',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
          onMouseOver={(e) => {
            e.target.style.background = '#5568d3';
          }}
          onMouseOut={(e) => {
            e.target.style.background = '#667eea';
          }}
        >
          Verify CAPTCHA
        </button>

        <button
          onClick={handleNext}
          disabled={!nextEnabled}
          style={{
            flex: 1,
            padding: '14px',
            fontSize: '16px',
            fontWeight: '600',
            color: nextEnabled ? 'white' : '#999',
            background: nextEnabled ? '#10b981' : '#e5e7eb',
            border: 'none',
            borderRadius: '8px',
            cursor: nextEnabled ? 'pointer' : 'not-allowed',
            transition: 'all 0.2s ease'
          }}
          onMouseOver={(e) => {
            if (nextEnabled) {
              e.target.style.background = '#059669';
            }
          }}
          onMouseOut={(e) => {
            if (nextEnabled) {
              e.target.style.background = '#10b981';
            }
          }}
        >
          Next
        </button>
      </div>

      {nextEnabled && (
        <p style={{ 
          textAlign: 'center', 
          color: '#10b981', 
          fontSize: '14px',
          marginTop: '10px'
        }}>
            CAPTCHA verified. You can proceed.
        </p>
      )}
    </div>
  );
}

