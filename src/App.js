import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SliderCaptchaPage from './SliderCaptchaPage';
import RotationCaptchaPage from './RotationCaptchaPage';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<SliderCaptchaPage />} />
        <Route path="/rotation-captcha" element={<RotationCaptchaPage />} />
      </Routes>
    </Router>
  );
}

export default App;
