import { useState, useRef, useEffect } from 'react';
import './CustomSliderCaptcha.css';

const CustomSliderCaptcha = ({ imageUrl, onVerify, onReset }) => {
  const [sliderPosition, setSliderPosition] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [isVerified, setIsVerified] = useState(false);
  const [isFailed, setIsFailed] = useState(false);
  const [puzzlePosition, setPuzzlePosition] = useState(0);
  const [imageLoaded, setImageLoaded] = useState(false);

  const sliderRef = useRef(null);
  const containerRef = useRef(null);
  const startXRef = useRef(0);

  // Generate random puzzle position when component mounts or image changes
  useEffect(() => {
    const randomX = Math.floor(Math.random() * 150) + 100; // Between 100-250px
    setPuzzlePosition(randomX);
  }, [imageUrl]);

  const handleMouseDown = (e) => {
    e.preventDefault();
    setIsDragging(true);
    startXRef.current = e.clientX - sliderPosition;
  };

  const handleTouchStart = (e) => {
    setIsDragging(true);
    startXRef.current = e.touches[0].clientX - sliderPosition;
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;

    const containerWidth = containerRef.current?.offsetWidth || 300;
    const maxSlide = containerWidth - 50; // 50px is slider button width
    const newPosition = Math.min(Math.max(0, e.clientX - startXRef.current), maxSlide);
    setSliderPosition(newPosition);
  };

  const handleTouchMove = (e) => {
    if (!isDragging) return;

    const containerWidth = containerRef.current?.offsetWidth || 300;
    const maxSlide = containerWidth - 50;
    const newPosition = Math.min(Math.max(0, e.touches[0].clientX - startXRef.current), maxSlide);
    setSliderPosition(newPosition);
  };

  const handleMouseUp = () => {
    if (!isDragging) return;
    setIsDragging(false);
    verifyPosition();
  };

  const handleTouchEnd = () => {
    if (!isDragging) return;
    setIsDragging(false);
    verifyPosition();
  };

  const verifyPosition = () => {
    const tolerance = 10;
    const isCorrect = Math.abs(sliderPosition - puzzlePosition) < tolerance;

    if (isCorrect) {
      setIsVerified(true);
      setIsFailed(false);
      onVerify({ success: true, position: sliderPosition, target: puzzlePosition });
    } else {
      setIsFailed(true);
      setTimeout(() => {
        setSliderPosition(0);
        setIsFailed(false);
      }, 1000);
      onVerify({ success: false, position: sliderPosition, target: puzzlePosition });
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
          {isVerified ? '' : isFailed ? 'Try again' : 'Slide to verify'}
        </div>
      </div>

      {isVerified && (
        <button className="reset-button" onClick={handleReset}>
          Reset
        </button>
      )}
    </div>
  );
};

export default CustomSliderCaptcha;
