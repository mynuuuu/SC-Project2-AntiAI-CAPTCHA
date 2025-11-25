import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SliderCaptchaPage from './SliderCaptchaPage';
import AnimalRotationPage from './AnimalRotationPage';
import AnimalSelectionPage from './AnimalSelectionPage';
import MorseCaptcha from './MorseCaptcha';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<SliderCaptchaPage />} />
        <Route path="/rotation-captcha" element={<AnimalRotationPage />} />
        <Route path="/animal-selection" element={<AnimalSelectionPage />} />
        <Route path="/morse-captcha" element={<MorseCaptcha />} />
      </Routes>
    </Router>
  );
}

export default App;
