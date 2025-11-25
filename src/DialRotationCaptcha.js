import React, { useState, useRef, useMemo, useEffect } from 'react';
import './DialRotationCaptcha.css';
import useBehaviorTracking from './useBehaviorTracking';
import { shouldCaptureBehavior } from './utils/behaviorMode';

const TOLERANCE = 15; // degrees tolerance for correct alignment
const CAPTCHA_ID = 'captcha2';

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

const createInitialInteractionStats = () => ({
  dragCount: 0,
  totalRotation: 0,
  directionChanges: 0,
  lastDeltaSign: 0,
  rotationTrace: [],
  usedMouse: false,
  usedTouch: false,
  startedAt: null,
  firstInteractionDelay: null,
  maxSpeed: 0,
  lastMoveTimestamp: null,
});

function DialRotationCaptcha({ onSuccess }) {
  const randomAnimal = useMemo(() => getRandomAnimalImage(), []);
  const [dialRotation, setDialRotation] = useState(0);
  // const [targetRotation] = useState(getRandomDirection());
  const [message, setMessage] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const dialRef = useRef(null);
  const lastAngleRef = useRef(0);
  const mountTimeRef = useRef(null);
  const interactionStatsRef = useRef(createInitialInteractionStats());
  const shouldLogBehavior = shouldCaptureBehavior();

  const {
    startRecording,
    stopRecording,
    captureEvent,
    sendToServer,
    getStats,
    isRecording,
    sessionId,
  } = useBehaviorTracking();

  useEffect(() => {
    mountTimeRef.current = performance.now();
  }, []);

  const resetInteractionStats = () => {
    interactionStatsRef.current = createInitialInteractionStats();
  };

  const pointerDownStats = (mode) => {
    const stats = interactionStatsRef.current;
    stats.dragCount += 1;
    if (mode === 'mouse') stats.usedMouse = true;
    if (mode === 'touch') stats.usedTouch = true;
    const now = performance.now();
    if (!stats.startedAt) {
      stats.startedAt = now;
      stats.firstInteractionDelay = mountTimeRef.current
        ? now - mountTimeRef.current
        : 0;
    }
  };

  const updateRotationStats = (deltaAngle, newRotation) => {
    if (deltaAngle === 0) return;
    const stats = interactionStatsRef.current;
    const now = performance.now();
    const deltaAbs = Math.abs(deltaAngle);
    stats.totalRotation += deltaAbs;
    const deltaSign = deltaAngle > 0 ? 1 : -1;
    if (stats.lastDeltaSign !== 0 && deltaSign !== stats.lastDeltaSign) {
      stats.directionChanges += 1;
    }
    stats.lastDeltaSign = deltaSign;

    if (stats.lastMoveTimestamp) {
      const elapsed = now - stats.lastMoveTimestamp;
      if (elapsed > 0) {
        const speed = deltaAbs / (elapsed / 1000);
        if (speed > stats.maxSpeed) {
          stats.maxSpeed = speed;
        }
      }
    }
    stats.lastMoveTimestamp = now;

    const relativeTime = stats.startedAt ? now - stats.startedAt : 0;
    stats.rotationTrace.push({
      t: Number(relativeTime.toFixed(2)),
      rotation: Number(newRotation.toFixed(2)),
    });
    if (stats.rotationTrace.length > 200) {
      stats.rotationTrace.shift();
    }
  };

  const toPointerLikeEvent = (touchEvent) => ({
    clientX: touchEvent.clientX,
    clientY: touchEvent.clientY,
    pageX: touchEvent.pageX,
    pageY: touchEvent.pageY,
    screenX: touchEvent.screenX,
    screenY: touchEvent.screenY,
    button: 0,
    buttons: 1,
    ctrlKey: false,
    shiftKey: false,
    altKey: false,
    metaKey: false,
  });

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
    if (!isRecording && shouldLogBehavior) {
      startRecording();
    }
    if (shouldLogBehavior) {
      captureEvent('mousedown', e, dialRef.current);
    }
    pointerDownStats('mouse');
  };

  const handleTouchStart = (e) => {
    setIsDragging(true);
    setMessage('');
    lastAngleRef.current = getAngleFromCenter(e, dialRef.current);
    if (!isRecording && shouldLogBehavior) {
      startRecording();
    }
    if (shouldLogBehavior) {
      const touchEvent = toPointerLikeEvent(e.touches[0]);
      captureEvent('touchstart', touchEvent, dialRef.current);
    }
    pointerDownStats('touch');
  };

  const handleMouseMove = (e) => {
    if (!isDragging || !dialRef.current) return;

    const currentAngle = getAngleFromCenter(e, dialRef.current);
    const deltaAngle = currentAngle - lastAngleRef.current;

    setDialRotation((prev) => {
      let newRotation = (prev + deltaAngle) % 360;
      if (newRotation < 0) newRotation += 360;
      updateRotationStats(deltaAngle, newRotation);
      return newRotation;
    });

    if (shouldLogBehavior) {
      captureEvent('mousemove', e, dialRef.current);
    }

    lastAngleRef.current = currentAngle;
  };

  const handleTouchMove = (e) => {
    if (!isDragging || !dialRef.current) return;

    const currentAngle = getAngleFromCenter(e, dialRef.current);
    const deltaAngle = currentAngle - lastAngleRef.current;

    setDialRotation((prev) => {
      let newRotation = (prev + deltaAngle) % 360;
      if (newRotation < 0) newRotation += 360;
      updateRotationStats(deltaAngle, newRotation);
      return newRotation;
    });

    if (shouldLogBehavior) {
      const touchEvent = toPointerLikeEvent(e.touches[0]);
      captureEvent('touchmove', touchEvent, dialRef.current);
    }

    lastAngleRef.current = currentAngle;
  };

  const handleMouseUp = (e) => {
    setIsDragging(false);
    if (shouldLogBehavior) {
      captureEvent('mouseup', e, dialRef.current);
    }
  };

  const handleTouchEnd = (e) => {
    setIsDragging(false);
    if (shouldLogBehavior) {
      const touchEvent = toPointerLikeEvent(e.changedTouches[0]);
      captureEvent('touchend', touchEvent, dialRef.current);
    }
  };

  const checkAlignment = async () => {
    const direction = getGuessedDirection(dialRotation);
    const isSuccess = direction === randomAnimal.name;

    // Stop recording and send data
    const events = shouldLogBehavior ? stopRecording() : [];
    const behaviorStats = shouldLogBehavior ? getStats() : {};
    const stats = interactionStatsRef.current;
    const now = performance.now();
    const interactionDuration = stats.startedAt ? now - stats.startedAt : 0;

    // Send data to server with rotation-specific metadata
    if (shouldLogBehavior && events.length > 0) {
      const metadata = {
        target_direction: randomAnimal.name,
        final_rotation_deg: Number(dialRotation.toFixed(2)),
        guessed_direction: direction,
        success: isSuccess,
        drag_count: stats.dragCount,
        total_rotation_deg: Number(stats.totalRotation.toFixed(2)),
        direction_changes: stats.directionChanges,
        max_rotation_speed_deg_per_sec: Number(stats.maxSpeed.toFixed(2)),
        interaction_duration_ms: Number(interactionDuration.toFixed(2)),
        idle_before_first_drag_ms: stats.firstInteractionDelay
          ? Number(stats.firstInteractionDelay.toFixed(2))
          : 0,
        used_mouse: stats.usedMouse,
        used_touch: stats.usedTouch,
        rotation_trace: stats.rotationTrace,
        behavior_event_count: events.length,
        behavior_stats: behaviorStats,
      };

      try {
        const serverUrl = 'http://localhost:5001/save_captcha_events';
        const response = await fetch(serverUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            captcha_id: CAPTCHA_ID,
            session_id: sessionId,
            captchaType: 'rotation',
            events: events,
            metadata: metadata,
            success: isSuccess,
          }),
        });

        const result = await response.json();
        
        if (result.success) {
          console.log(`✓ Rotation data saved to ${CAPTCHA_ID}.csv`);
        } else {
          console.error('Failed to save data:', result.error);
        }
      } catch (error) {
        console.error('Error saving data:', error);
      }
    } else if (!shouldLogBehavior) {
      console.log('Behavior logging skipped (not running via npm start).');
    }

    resetInteractionStats();

    // Normal captcha verification flow
    if (isSuccess) {
      setMessage('✅ Captcha Solved');
      if (onSuccess) {
        setTimeout(onSuccess, 1000);
      }
    } else {
      setMessage('❌ Try Again!');
    }
  };

  const handleSkip = () => {
    setMessage('⏭️ Skipped');
    if (onSuccess) {
      setTimeout(onSuccess, 500); // Navigate to next captcha
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

      {isRecording && shouldLogBehavior && (
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
        <button
          onClick={handleSkip}
          className="dial-captcha-button-skip"
        >
          Skip
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
