import React from 'react';
import { useNavigate } from 'react-router-dom';

export default function BotDetectedPage() {
  const navigate = useNavigate();

  const handleGoBack = () => {
    // Clear any stored flags
    localStorage.removeItem('captcha_passed');
    localStorage.removeItem('is_human');
    
    // Go back to entry form
    navigate('/entry');
  };

  return (
    <div style={{
      textAlign: 'center',
      marginTop: '100px',
      padding: '40px'
    }}>
      <div style={{
        maxWidth: '600px',
        margin: '0 auto',
        padding: '40px',
        border: '3px solid #ef4444',
        borderRadius: '12px',
        backgroundColor: '#fef2f2',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
      }}>
        <h1 style={{ 
          color: '#ef4444', 
          fontSize: '48px',
          marginBottom: '20px'
        }}>
          🚫
        </h1>
        <h2 style={{ 
          color: '#dc2626',
          marginBottom: '20px'
        }}>
          Bot Detected
        </h2>
        <p style={{ 
          fontSize: '18px',
          color: '#7f1d1d',
          marginBottom: '30px',
          lineHeight: '1.6'
        }}>
          Your behavior has been classified as automated/bot-like.
          <br />
          Access has been denied.
        </p>
        <button
          onClick={handleGoBack}
          style={{
            padding: '14px 32px',
            fontSize: '16px',
            fontWeight: '600',
            color: 'white',
            background: '#ef4444',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
          onMouseOver={(e) => {
            e.target.style.background = '#dc2626';
          }}
          onMouseOut={(e) => {
            e.target.style.background = '#ef4444';
          }}
        >
          Return to Login
        </button>
      </div>
    </div>
  );
}

