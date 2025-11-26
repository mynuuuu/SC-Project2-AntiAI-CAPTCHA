import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import EntryForm from './EntryForm';
import SliderCaptchaPage from './SliderCaptchaPage';
import AnimalRotationPage from './AnimalRotationPage';
import AnimalSelectionPage from './AnimalSelectionPage';
import MorseCaptcha from './MorseCaptcha';
import BotDetectedPage from './BotDetectedPage';
import HomePage from './HomePage';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        {/* Default entry point - redirect root to /entry */}
        <Route path="/" element={<Navigate to="/entry" replace />} />
        <Route path="/entry" element={<EntryForm />} />
        <Route path="/slider-captcha" element={<SliderCaptchaPage />} />
        <Route path="/rotation-captcha" element={<AnimalRotationPage />} />
        <Route path="/animal-selection" element={<AnimalSelectionPage />} />
        <Route path="/morse-captcha" element={<MorseCaptcha />} />
        <Route path="/bot-detected" element={<BotDetectedPage />} />
        <Route path="/home" element={<HomePage />} />
      </Routes>
    </Router>
  );
}

export default App;
