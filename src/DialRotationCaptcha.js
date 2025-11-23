import React, { useState, useRef, useMemo } from 'react';
import './DialRotationCaptcha.css';

const TOLERANCE = 10; // degrees tolerance for correct alignment

// Available animal images with their corresponding directions
const animalImages = [
  { path: '/Rotation Animals/north.png', name: 'North' },
  { path: '/Rotation Animals/north east.png', name: 'North East' },
  { path: '/Rotation Animals/east.png', name: 'East' },
  { path: '/Rotation Animals/south east.png', name: 'South East' },
  { path: '/Rotation Animals/south.png', name: 'South' },
  { path: '/Rotation Animals/south west.png', name: 'South West' },
  { path: '/Rotation Animals/west.png', name: 'West' },
  { path: '/Rotation Animals/north west.png', name: 'North West' },
];

// const getRandomDirection = () => {
//   const directionsArr = Object.values(directions);
//   return directionsArr[Math.floor(Math.random() * directionsArr.length)];
// };

const getRandomAnimalImage = () => {
  return animalImages[Math.floor(Math.random() * animalImages.length)];
};

function DialRotationCaptcha({ onSuccess }) {
  const randomAnimal = useMemo(() => getRandomAnimalImage(), []);
  const [dialRotation, setDialRotation] = useState(0);
  // const [targetRotation] = useState(getRandomDirection());
  const [message, setMessage] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const dialRef = useRef(null);
  const lastAngleRef = useRef(0);

  const getAngleFromCenter = (e, element) => {
    const rect = element.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    const clientX = e.type.includes('touch') ? e.touches[0].clientX : e.clientX;
    const clientY = e.type.includes('touch') ? e.touches[0].clientY : e.clientY;

    const deltaX = clientX - centerX;
    const deltaY = clientY - centerY;

    // Calculate angle in degrees (0° is up, clockwise)
    let angle = Math.atan2(deltaX, -deltaY) * (180 / Math.PI);
    if (angle < 0) angle += 360;

    return angle;
  };

  const getGuessedDirection = (degree) => {
    switch (true) {
      case (degree > 350 || degree < 10):
        return 'North';

      case (degree > 35 && degree < 55):
        return 'North East';

      case (degree > 80 && degree < 100):
        return 'East';

      case (degree > 125 && degree < 145):
        return 'South East';

      case (degree > 170 && degree < 190):
        return 'South';

      case (degree > 215 && degree < 235):
        return 'South West';

      case (degree > 260 && degree < 280):
        return 'West';

      case (degree > 305 && degree < 325):
        return 'North West';

      default:
        return 'Unknown';
    }
  };

  const handleMouseDown = (e) => {
    e.preventDefault();
    setIsDragging(true);
    setMessage('');
    lastAngleRef.current = getAngleFromCenter(e, dialRef.current);
  };

  const handleTouchStart = (e) => {
    setIsDragging(true);
    setMessage('');
    lastAngleRef.current = getAngleFromCenter(e, dialRef.current);
  };

  const handleMouseMove = (e) => {
    if (!isDragging || !dialRef.current) return;

    const currentAngle = getAngleFromCenter(e, dialRef.current);
    const deltaAngle = currentAngle - lastAngleRef.current;

    setDialRotation((prev) => {
      let newRotation = (prev + deltaAngle) % 360;
      if (newRotation < 0) newRotation += 360;
      return newRotation;
    });

    lastAngleRef.current = currentAngle;
  };

  const handleTouchMove = (e) => {
    if (!isDragging || !dialRef.current) return;

    const currentAngle = getAngleFromCenter(e, dialRef.current);
    const deltaAngle = currentAngle - lastAngleRef.current;

    setDialRotation((prev) => {
      let newRotation = (prev + deltaAngle) % 360;
      if (newRotation < 0) newRotation += 360;
      return newRotation;
    });

    lastAngleRef.current = currentAngle;
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleTouchEnd = () => {
    setIsDragging(false);
  };

  const checkAlignment = () => {
    const direction = getGuessedDirection(dialRotation);

    if (direction === randomAnimal.name) {
      setMessage('✅ Captcha Passed!');
      if (onSuccess) {
        setTimeout(onSuccess, 1000); // Delay to show success message
      }
    } else {
      setMessage('❌ Try Again!');
    }
  };

  // Add event listeners
  React.useEffect(() => {
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
  }, [isDragging]);

  return (
    <div className="dial-rotation-captcha-container">
      <h2 className="dial-rotation-captcha-title">
        Rotate the dial to the direction the animal is facing
      </h2>

      <div className="dial-container">
        {/* Target animal indicator */}
        {/* <div className="target-indicator"> */}
        {/* <div className="target-label">Target Direction: {getTargetDirectionName()}</div> */}
        <div className="target-animal-wrapper">
          <img
            src={randomAnimal.path}
            alt={randomAnimal.name}
            className="target-animal"
          />
        </div>
        {/* </div> */}

        {/* Dial */}
        <div
          ref={dialRef}
          className={`dial ${isDragging ? 'dragging' : ''}`}
          style={{ transform: `rotate(${dialRotation}deg)` }}
          onMouseDown={handleMouseDown}
          onTouchStart={handleTouchStart}
        >
          {/* Dial markers */}
          <div className="dial-center"></div>
          <div className="dial-pointer">↑</div>
          {/* <div className="dial-marker marker-0">0°</div>
          <div className="dial-marker marker-90">90°</div>
          <div className="dial-marker marker-180">180°</div>
          <div className="dial-marker marker-270">270°</div> */}
        </div>

        {/* Degree display */}
        <div className="degree-display">
          {Math.round(dialRotation)}°
        </div>
      </div>

      <div className="dial-controls">
        <button
          onClick={checkAlignment}
          className="dial-captcha-button-submit"
        >
          Submit
        </button>
      </div>

      {message && (
        <p className={`dial-captcha-message ${message.includes('✅')
          ? 'dial-captcha-message-success'
          : 'dial-captcha-message-error'
          }`}>
          {message}
        </p>
      )}
    </div>
  );
}

export default DialRotationCaptcha;
