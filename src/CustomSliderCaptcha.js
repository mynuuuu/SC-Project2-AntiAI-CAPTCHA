import { useState, useRef, useEffect } from 'react';
import './CustomSliderCaptcha.css';
import useBehaviorTracking from './useBehaviorTracking';

const CustomSliderCaptcha = ({ imageUrl, onVerify, onReset, captchaId }) => {
  const [sliderPosition, setSliderPosition] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [isVerified, setIsVerified] = useState(false);
  const [isFailed, setIsFailed] = useState(false);
  const [puzzlePosition, setPuzzlePosition] = useState(0);
  const [imageLoaded, setImageLoaded] = useState(false);

  const sliderRef = useRef(null);
  const containerRef = useRef(null);
  const startXRef = useRef(0);

  // Behavior tracking hook
  const {
    startRecording,
    stopRecording,
    captureEvent,
    saveToCSV,
    sendToServer,
    getStats,
    isRecording,
  } = useBehaviorTracking();

  // Generate random puzzle position when component mounts or image changes
  useEffect(() => {
    const randomX = Math.floor(Math.random() * 150) + 100; // Between 100-250px
    setPuzzlePosition(randomX);
  }, [imageUrl]);

  const handleMouseDown = (e) => {
    e.preventDefault();
    setIsDragging(true);
    startXRef.current = e.clientX - sliderPosition;

    // Start behavior recording
    startRecording();
    captureEvent('mousedown', e, containerRef.current);
  };

  const handleTouchStart = (e) => {
    setIsDragging(true);
    startXRef.current = e.touches[0].clientX - sliderPosition;

    // Start behavior recording for touch
    startRecording();
    // Convert touch event to mouse-like event for tracking
    const touchEvent = {
      clientX: e.touches[0].clientX,
      clientY: e.touches[0].clientY,
      pageX: e.touches[0].pageX,
      pageY: e.touches[0].pageY,
      screenX: e.touches[0].screenX,
      screenY: e.touches[0].screenY,
    };
    captureEvent('touchstart', touchEvent, containerRef.current);
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;

    const containerWidth = containerRef.current?.offsetWidth || 300;
    const maxSlide = containerWidth - 50; // 50px is slider button width
    const newPosition = Math.min(Math.max(0, e.clientX - startXRef.current), maxSlide);
    setSliderPosition(newPosition);

    // Capture mouse movement
    captureEvent('mousemove', e, containerRef.current);
  };

  const handleTouchMove = (e) => {
    if (!isDragging) return;

    const containerWidth = containerRef.current?.offsetWidth || 300;
    const maxSlide = containerWidth - 50;
    const newPosition = Math.min(Math.max(0, e.touches[0].clientX - startXRef.current), maxSlide);
    setSliderPosition(newPosition);

    // Capture touch movement
    const touchEvent = {
      clientX: e.touches[0].clientX,
      clientY: e.touches[0].clientY,
      pageX: e.touches[0].pageX,
      pageY: e.touches[0].pageY,
      screenX: e.touches[0].screenX,
      screenY: e.touches[0].screenY,
    };
    captureEvent('touchmove', touchEvent, containerRef.current);
  };

  const handleMouseUp = (e) => {
    if (!isDragging) return;
    setIsDragging(false);

    // Capture mouse up event
    captureEvent('mouseup', e, containerRef.current);
    verifyPosition();
  };

  const handleTouchEnd = (e) => {
    if (!isDragging) return;
    setIsDragging(false);

    // Capture touch end event
    const touchEvent = {
      clientX: e.changedTouches[0].clientX,
      clientY: e.changedTouches[0].clientY,
      pageX: e.changedTouches[0].pageX,
      pageY: e.changedTouches[0].pageY,
      screenX: e.changedTouches[0].screenX,
      screenY: e.changedTouches[0].screenY,
    };
    captureEvent('touchend', touchEvent, containerRef.current);
    verifyPosition();
  };

  const verifyPosition = async () => {
    const tolerance = 10;
    const isCorrect = Math.abs(sliderPosition - puzzlePosition) < tolerance;

    // Stop recording
    const events = stopRecording();
    const stats = getStats();

    // Send data to server
    const serverUrl = 'http://localhost:5001/save_captcha_events';
    const result = await sendToServer(serverUrl, 'human', captchaId, isCorrect);
    
    if (result.success) {
      console.log(`✓ Data saved to ${captchaId}.csv`);
    } else {
      console.error('Failed to save data to server:', result.error);
    }

    if (isCorrect) {
      setIsVerified(true);
      setIsFailed(false);

      console.log('Captcha solved! Behavior data saved to server.');
      console.log('Stats:', stats);

      onVerify({
        success: true,
        position: sliderPosition,
        target: puzzlePosition,
        behaviorStats: stats,
        eventCount: events.length
      });

      // Don't auto-reset on success - keep it verified
    } else {
      setIsFailed(true);
      
      console.log('Captcha failed. Behavior data saved to server.');
      console.log('Stats:', stats);
      
      onVerify({ 
        success: false, 
        position: sliderPosition, 
        target: puzzlePosition,
        behaviorStats: stats,
        eventCount: events.length
      });
      
      // Auto-reset after 1 second
      setTimeout(() => {
        setSliderPosition(0);
        setIsFailed(false);
      }, 1000);
    }
  };

  const handleReset = () => {
    setSliderPosition(0);
    setIsVerified(false);
    setIsFailed(false);
    const randomX = Math.floor(Math.random() * 150) + 100;
    setPuzzlePosition(randomX);
    if (onReset) onReset();
  };

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.addEventListener('touchmove', handleTouchMove);
      document.addEventListener('touchend', handleTouchEnd);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('touchmove', handleTouchMove);
      document.removeEventListener('touchend', handleTouchEnd);
    };
  }, [isDragging, sliderPosition]);

  return (
    <div className="custom-slider-captcha">
      {isRecording && (
        <div style={{
          position: 'absolute',
          top: '10px',
          right: '10px',
          background: 'rgba(255, 0, 0, 0.8)',
          color: 'white',
          padding: '5px 10px',
          borderRadius: '5px',
          fontSize: '12px',
          fontWeight: 'bold',
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          gap: '5px'
        }}>
          <span style={{
            width: '8px',
            height: '8px',
            backgroundColor: 'white',
            borderRadius: '50%',
            animation: 'pulse 1s infinite'
          }}></span>
          Recording
        </div>
      )}
      <div className="captcha-image-container" ref={containerRef}>
        <img
          src={imageUrl}
          alt="Captcha"
          className="captcha-bg-image"
          onLoad={() => setImageLoaded(true)}
        />
        {imageLoaded && (
          <>
            {/* Puzzle cutout */}
            <div
              className="puzzle-cutout"
              style={{
                left: `${puzzlePosition}px`,
                opacity: isVerified ? 0 : 1.0
              }}
            />
            {/* Moving puzzle piece */}
            <div
              className={`puzzle-piece ${isVerified ? 'verified' : ''} ${isFailed ? 'failed' : ''}`}
              style={{
                left: `${sliderPosition}px`,
                backgroundImage: `url(${imageUrl})`,
                backgroundPosition: `-${isVerified ? sliderPosition : puzzlePosition}px -75px`,
              }}
            />
          </>
        )}
      </div>

      <div className={`slider-track ${isVerified ? 'verified' : ''} ${isFailed ? 'failed' : ''}`}>
        <div className="slider-fill" style={{ width: `${sliderPosition}px` }} />
        <div
          className={`slider-button ${isDragging ? 'dragging' : ''}`}
          style={{ left: `${sliderPosition}px` }}
          onMouseDown={handleMouseDown}
          onTouchStart={handleTouchStart}
        >
          {isVerified ? '✓' : isFailed ? '✗' : '→'}
        </div>
        <div className="slider-text">
          {isVerified ? '' : isFailed ? 'Try again...' : 'Slide to verify'}
        </div>
      </div>
    </div>
  );
};

export default CustomSliderCaptcha;
