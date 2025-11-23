import React, { useState } from 'react';
import './AnimalRotationCaptcha.css';

const TOLERANCE = 15;

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

  const rotateAnimal = (delta) => {
    setAnimalRotation((prev) => (prev + delta + 360) % 360);
    setMessage('');
  };

  const checkAlignment = () => {
    const diff = Math.abs(animalRotation - targetRotation);
    if (diff <= TOLERANCE || diff >= 360 - TOLERANCE) {
      setMessage('✅ Captcha Passed!');
      if (onSuccess) {
        onSuccess();
      }
    } else {
      setMessage('❌ Try Again!');
    }
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
            onClick={() => rotateAnimal(-15)}
            className="rotation-captcha-button"
          >
            ←
          </button>
          <button
            onClick={() => rotateAnimal(15)}
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
        <p className={`rotation-captcha-message ${
          message.includes('✅')
            ? 'rotation-captcha-message-success'
            : 'rotation-captcha-message-error'
        }`}>
          {message}
        </p>
      )}
    </div>
  );
}

export default AnimalRotationCaptcha;
