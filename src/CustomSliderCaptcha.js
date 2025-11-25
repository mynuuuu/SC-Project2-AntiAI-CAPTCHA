import { useState, useRef, useEffect } from 'react';
import './CustomSliderCaptcha.css';
import useBehaviorTracking from './useBehaviorTracking';
import { shouldCaptureBehavior } from './utils/behaviorMode';

const createInitialInteractionStats = () => ({
  dragCount: 0,
  totalTravel: 0,
  directionChanges: 0,
  lastDeltaSign: 0,
  usedMouse: false,
  usedTouch: false,
  startedAt: null,
  firstInteractionDelay: null,
  maxSpeed: 0,
  lastMoveTimestamp: null,
  sliderTrace: [],
});

const CustomSliderCaptcha = ({ imageUrl, onVerify, onReset, captchaId }) => {
  const [sliderPosition, setSliderPosition] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [isVerified, setIsVerified] = useState(false);
  const [isFailed, setIsFailed] = useState(false);
  const [puzzlePosition, setPuzzlePosition] = useState(0);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [message, setMessage] = useState('');

  const sliderRef = useRef(null);
  const containerRef = useRef(null);
  const startXRef = useRef(0);
  const sliderPositionRef = useRef(0);
  const mountTimeRef = useRef(null);
  const interactionStatsRef = useRef(createInitialInteractionStats());

  // Behavior tracking hook
  const {
    startRecording,
    stopRecording,
    captureEvent,
    sendToServer,
    getStats,
    isRecording,
    sessionId,
  } = useBehaviorTracking();
  const shouldLogBehavior = shouldCaptureBehavior();
  const resolvedCaptchaId = captchaId || 'captcha1';

  useEffect(() => {
    mountTimeRef.current = performance.now();
  }, []);

  // Generate random puzzle position when component mounts or image changes
  useEffect(() => {
    const randomX = Math.floor(Math.random() * 150) + 100; // Between 100-250px
    setPuzzlePosition(randomX);
  }, [imageUrl]);

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

  const updateMoveStats = (delta, newPosition) => {
    if (delta === 0) return;
    const stats = interactionStatsRef.current;
    const now = performance.now();
    const deltaAbs = Math.abs(delta);
    stats.totalTravel += deltaAbs;
    const deltaSign = delta > 0 ? 1 : -1;
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
    stats.sliderTrace.push({
      t: Number(relativeTime.toFixed(2)),
      position: Number(newPosition.toFixed(2)),
    });
    if (stats.sliderTrace.length > 200) {
      stats.sliderTrace.shift();
    }
  };

  const handleMouseDown = (e) => {
    e.preventDefault();
    setIsDragging(true);
    startXRef.current = e.clientX - sliderPosition;
    sliderPositionRef.current = sliderPosition;

    // Start behavior recording
    if (!isRecording && shouldLogBehavior) {
      startRecording();
    }
    if (shouldLogBehavior) {
      captureEvent('mousedown', e, containerRef.current);
    }
    pointerDownStats('mouse');
  };

  const handleTouchStart = (e) => {
    setIsDragging(true);
    startXRef.current = e.touches[0].clientX - sliderPosition;
    sliderPositionRef.current = sliderPosition;

    // Start behavior recording for touch
    if (!isRecording && shouldLogBehavior) {
      startRecording();
    }
    if (shouldLogBehavior) {
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
    }
    pointerDownStats('touch');
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;

    const containerWidth = containerRef.current?.offsetWidth || 300;
    const maxSlide = containerWidth - 50; // 50px is slider button width
    const prevPosition = sliderPositionRef.current;
    const newPosition = Math.min(Math.max(0, e.clientX - startXRef.current), maxSlide);
    const delta = newPosition - prevPosition;
    setSliderPosition(newPosition);
    sliderPositionRef.current = newPosition;
    updateMoveStats(delta, newPosition);

    // Capture mouse movement
    if (shouldLogBehavior) {
      captureEvent('mousemove', e, containerRef.current);
    }
  };

  const handleTouchMove = (e) => {
    if (!isDragging) return;

    const containerWidth = containerRef.current?.offsetWidth || 300;
    const maxSlide = containerWidth - 50;
    const prevPosition = sliderPositionRef.current;
    const newPosition = Math.min(Math.max(0, e.touches[0].clientX - startXRef.current), maxSlide);
    const delta = newPosition - prevPosition;
    setSliderPosition(newPosition);
    sliderPositionRef.current = newPosition;
    updateMoveStats(delta, newPosition);

    // Capture touch movement
    if (shouldLogBehavior) {
      const touchEvent = {
        clientX: e.touches[0].clientX,
        clientY: e.touches[0].clientY,
        pageX: e.touches[0].pageX,
        pageY: e.touches[0].pageY,
        screenX: e.touches[0].screenX,
        screenY: e.touches[0].screenY,
      };
      captureEvent('touchmove', touchEvent, containerRef.current);
    }
  };

  const handleMouseUp = (e) => {
    if (!isDragging) return;
    setIsDragging(false);

    // Capture mouse up event
    if (shouldLogBehavior) {
      captureEvent('mouseup', e, containerRef.current);
    }
    verifyPosition();
  };

  const handleTouchEnd = (e) => {
    if (!isDragging) return;
    setIsDragging(false);

    // Capture touch end event
    if (shouldLogBehavior) {
      const touchEvent = {
        clientX: e.changedTouches[0].clientX,
        clientY: e.changedTouches[0].clientY,
        pageX: e.changedTouches[0].pageX,
        pageY: e.changedTouches[0].pageY,
        screenX: e.changedTouches[0].screenX,
        screenY: e.changedTouches[0].screenY,
      };
      captureEvent('touchend', touchEvent, containerRef.current);
    }
    verifyPosition();
  };

  const verifyPosition = async () => {
    const tolerance = 10;
    const isCorrect = Math.abs(sliderPosition - puzzlePosition) < tolerance;

    // Stop recording and get stats
    const events = shouldLogBehavior ? stopRecording() : [];
    const behaviorStats = shouldLogBehavior ? getStats() : {};
    const stats = interactionStatsRef.current;
    const now = performance.now();
    const interactionDuration = stats.startedAt ? now - stats.startedAt : 0;

    // Send data to server with slider-specific metadata
    if (shouldLogBehavior && events.length > 0) {
      const metadata = {
        target_position_px: Number(puzzlePosition.toFixed(2)),
        final_slider_position_px: Number(sliderPosition.toFixed(2)),
        success: isCorrect,
        drag_count: stats.dragCount,
        total_travel_px: Number(stats.totalTravel.toFixed(2)),
        direction_changes: stats.directionChanges,
        max_speed_px_per_sec: Number(stats.maxSpeed.toFixed(2)),
        interaction_duration_ms: Number(interactionDuration.toFixed(2)),
        idle_before_first_drag_ms: stats.firstInteractionDelay
          ? Number(stats.firstInteractionDelay.toFixed(2))
          : 0,
        used_mouse: stats.usedMouse,
        used_touch: stats.usedTouch,
        slider_trace: stats.sliderTrace,
        behavior_event_count: events.length,
        behavior_stats: behaviorStats,
      };

      try {
        const serverUrl = 'http://localhost:5001/save_captcha_events';
        const response = await fetch(serverUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            captcha_id: resolvedCaptchaId,
            session_id: sessionId,
            captchaType: 'slider',
            events: events,
            metadata: metadata,
            success: isCorrect,
          }),
        });

        const result = await response.json();
        
        if (result.success) {
          console.log(`✓ Data saved to ${resolvedCaptchaId}.csv`);
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
    if (isCorrect) {
      setIsVerified(true);
      setIsFailed(false);
      setMessage('✅ Captcha Solved');

      onVerify({
        success: true,
        position: sliderPosition,
        target: puzzlePosition,
        behaviorStats: stats,
        eventCount: events.length,
      });
    } else {
      setIsFailed(true);
      setMessage('❌ Try Again');

      onVerify({
        success: false,
        position: sliderPosition,
        target: puzzlePosition,
        behaviorStats: stats,
        eventCount: events.length,
      });

      // Auto-reset after 1 second
      setTimeout(() => {
        setSliderPosition(0);
        setIsFailed(false);
        setMessage('');
      }, 1000);
    }
  };

  const handleReset = () => {
    setSliderPosition(0);
    sliderPositionRef.current = 0;
    setIsVerified(false);
    setIsFailed(false);
    const randomX = Math.floor(Math.random() * 150) + 100;
    setPuzzlePosition(randomX);
    resetInteractionStats();
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
