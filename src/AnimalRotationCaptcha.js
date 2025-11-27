import React, { useRef, useState } from 'react';
import './AnimalRotationCaptcha.css';
import useBehaviorTracking from './useBehaviorTracking';
import { shouldCaptureBehavior } from './utils/behaviorMode';
import { BEHAVIOR_API } from './config/apiConfig';

const TOLERANCE = 15;
const CAPTCHA_TYPE = 'rotation';
const CAPTCHA_ID = 'captcha2';

const directions = {
  UP: 0,
  RIGHT: 90,
  DOWN: 180,
  LEFT: 270,
};

const getRandomDirection = () => {
  const directionsArr = Object.values(directions);
  return directionsArr[Math.floor(Math.random() * directionsArr.length)];
};

function AnimalRotationCaptcha({ onSuccess }) {
  const [animalRotation, setAnimalRotation] = useState(getRandomDirection());
  const [targetRotation] = useState(getRandomDirection());
  const [message, setMessage] = useState('');
  const containerRef = useRef(null);
  const shouldLogBehavior = shouldCaptureBehavior();

  const {
    startRecording,
    stopRecording,
    captureEvent,
    sendToServer,
    getStats,
    isRecording,
    reset,
  } = useBehaviorTracking();

  const ensureRecording = () => {
    if (!isRecording) {
      startRecording();
    }
  };

  const normalizeEvent = (event) => {
    if (!event) return null;

    if (event.touches && event.touches[0]) {
      const touch = event.touches[0];
      return {
        clientX: touch.clientX,
        clientY: touch.clientY,
        pageX: touch.pageX,
        pageY: touch.pageY,
        screenX: touch.screenX,
        screenY: touch.screenY,
        button: 0,
        buttons: 1,
      };
    }

    if (event.changedTouches && event.changedTouches[0]) {
      const touch = event.changedTouches[0];
      return {
        clientX: touch.clientX,
        clientY: touch.clientY,
        pageX: touch.pageX,
        pageY: touch.pageY,
        screenX: touch.screenX,
        screenY: touch.screenY,
        button: 0,
        buttons: 0,
      };
    }

    return event;
  };

  const recordEvent = (eventType, reactEvent, autoStart = true) => {
    const nativeEvent = reactEvent?.nativeEvent || reactEvent;

    if (autoStart) {
      ensureRecording();
    } else if (!isRecording) {
      return;
    }

    const normalizedEvent = normalizeEvent(nativeEvent);
    if (normalizedEvent) {
      captureEvent(eventType, normalizedEvent, containerRef.current);
    }
  };

  const rotateAnimal = (delta, event) => {
    if (event?.preventDefault) {
      event.preventDefault();
    }

    recordEvent(delta > 0 ? 'rotate_right' : 'rotate_left', event);
    setAnimalRotation((prev) => (prev + delta + 360) % 360);
    setMessage('');
  };

  const handleMouseMove = (event) => {
    recordEvent('mousemove', event, false);
  };

  const checkAlignment = async (event) => {
    if (event) {
      recordEvent('submit_click', event);
    }

    const diff = Math.abs(animalRotation - targetRotation);
    const isSuccess = diff <= TOLERANCE || diff >= 360 - TOLERANCE;
    setMessage(isSuccess ? '  Captcha Passed!' : '  Try Again!');

    let events = [];
    if (isRecording) {
      events = stopRecording();
    }
    const stats = getStats();

    if (shouldLogBehavior && events.length > 0) {
      const serverUrl = BEHAVIOR_API.saveCaptchaEvents;
      const result = await sendToServer(
        serverUrl,
        CAPTCHA_TYPE,
        'human',
        CAPTCHA_ID,
        isSuccess
      );
      if (result.success) {
        console.log('  Data saved to captcha2.csv');
      } else {
        console.error('Failed to save rotation captcha data:', result.error);
      }
    } else if (!shouldLogBehavior) {
      console.log('Behavior logging skipped for rotation captcha (human-only mode).');
    } else {
      console.warn('No rotation behavior events captured; skipping server upload.');
    }

    if (isSuccess && typeof onSuccess === 'function') {
      onSuccess(stats);
    }

    reset();
  };

  return (
    <div
      className="rotation-captcha-container"
      ref={containerRef}
      onMouseMove={handleMouseMove}
      onTouchMove={(event) => recordEvent('touchmove', event, false)}
    >
      <h2 className="rotation-captcha-title">
        Rotate the animal to face the finger
      </h2>

      <div className="rotation-captcha-images-container">
        <img
          src="/captcha_images/hand.png"
          alt="Finger"
          className="rotation-captcha-target"
          style={{ transform: `translateX(-50%) rotate(${targetRotation}deg)` }}
        />

        <div className="rotation-captcha-animal-wrapper">
          <img
            src="/captcha_images/rotation_captcha_1.png"
            alt="Animal"
            className="rotation-captcha-animal"
            style={{ transform: `rotate(${animalRotation}deg)` }}
          />
        </div>
      </div>

      <div className="rotation-captcha-controls">
        <div className="rotation-captcha-buttons">
          <button
            onMouseDown={(event) => rotateAnimal(-15, event)}
            onTouchStart={(event) => rotateAnimal(-15, event)}
            className="rotation-captcha-button"
          >
            ←
          </button>
          <button
            onMouseDown={(event) => rotateAnimal(15, event)}
            onTouchStart={(event) => rotateAnimal(15, event)}
            className="rotation-captcha-button"
          >
            →
          </button>
        </div>
        <button
          onMouseDown={(event) => recordEvent('submit_button_mousedown', event)}
          onTouchStart={(event) => recordEvent('submit_button_touchstart', event)}
          onClick={checkAlignment}
          className="rotation-captcha-button-submit"
        >
          Submit
        </button>
      </div>

      {message && (
        <p
          className={
            'rotation-captcha-message ' +
            (message.includes(' ')
              ? 'rotation-captcha-message-success'
              : 'rotation-captcha-message-error')
          }
        >
          {message}
        </p>
      )}
    </div>
  );
}

export default AnimalRotationCaptcha;
