import React, { useState } from 'react';
import './AnimalRotationCaptcha.css';
import useBehaviorTracking from './useBehaviorTracking';

const TOLERANCE = 15;
const captchaType = "rotation";
const captchaId = "rotation1";

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

  const {
    isRecording,
    startRecording,
    stopRecording,
    captureEvent,
    sendToServer,
  } = useBehaviorTracking();

  const handleRotate = (delta) => (e) => {
    // start tracking on first interaction
    if (!isRecording) {
      startRecording();
    }

    // update rotation + log event with rich info
    setAnimalRotation((prev) => {
      const next = (prev + delta + 360) % 360;

      captureEvent(
        delta > 0 ? 'rotate_right' : 'rotate_left',
        e,
        e.currentTarget,
        {
          rotation_delta: delta,
          rotation_angle: next,
          target_angle: targetRotation,
          captcha_id: captchaId,
          captcha_type: captchaType,
        }
      );

      setMessage('');
      return next;
    });
  };

  const checkAlignment = async () => {
    const diff = Math.abs(animalRotation - targetRotation);
    const passed = diff <= TOLERANCE || diff >= 360 - TOLERANCE;

    setMessage(passed ? '✅ Captcha Passed!' : '❌ Try Again!');

    // stop tracking and send behaviour to backend
    stopRecording();

    await sendToServer(
      'http://localhost:5001/save_captcha_events',
      captchaType,   // captchaType
      'human',        // userType
      captchaId,      // captcha_id
      passed          // success (boolean)
    );

    return passed;
  };

  return (
    <div className="rotation-captcha-container">
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
            onClick={handleRotate(-15)}
            className="rotation-captcha-button"
          >
            ←
          </button>
          <button
            onClick={handleRotate(15)}
            className="rotation-captcha-button"
          >
            →
          </button>
        </div>
        <button
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
            (message.includes('✅')
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
