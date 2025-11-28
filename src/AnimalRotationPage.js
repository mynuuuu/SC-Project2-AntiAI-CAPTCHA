import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import DialRotationCaptcha from './DialRotationCaptcha';
import './App.css';

// Available animal images for flying animation
export const animalImages = [
  {
    img: '/Flying Animals/Turtle Animal 3D Style.png',
    name: 'Turtle'
  },
  {
    img: '/Flying Animals/Pink Flamingo Wildlife PNG.png',
    name: 'Flamingo'
  },
  {
    img: '/Flying Animals/Panda Eating Bamboo Illustration.png',
    name: 'Panda'
  },
  {
    img: '/Flying Animals/Chipmunk Cartoon Design.png',
    name: 'Chipmunk'
  },
  {
    img: '/Flying Animals/Golden Brown Chicken Portrait.png',
    name: 'Chicken'
  }
];

function AnimalRotationPage() {
  const navigate = useNavigate();
  const [showAnimal, setShowAnimal] = useState(false);

  // Random configuration for the single animal (generated once)
  const animalConfig = useState(() => ({
    animal: animalImages[Math.floor(Math.random() * animalImages.length)],
    y: Math.random() * 60 + 10, 
    duration: Math.random() * 3 + 5, 
    size: Math.random() * 40 + 80, 
    delay: Math.random() * 2 + 1,
  }))[0];

  const handleBack = () => {
    navigate('/');
  };

  // Trigger the single flying animal animation
  useEffect(() => {
    // Start animation after random delay (3-8 seconds)
    const timeout = setTimeout(() => {
      setShowAnimal(true);
    }, animalConfig.delay * 1000);

    return () => clearTimeout(timeout);
  }, [animalConfig.delay]);

  return (
    <div className="App" style={{ position: 'relative', overflow: 'hidden' }}>
      {/* Single flying animal */}
      <AnimatePresence>
        {showAnimal && (
          <motion.img
            src={animalConfig.animal.img}
            alt="Flying animal"
            initial={{ x: `-${animalConfig.size + 50}px` }}
            animate={{ x: `calc(100vw + ${animalConfig.size + 50}px)` }}
            transition={{
              duration: animalConfig.duration,
              ease: 'linear',
            }}
            style={{
              position: 'fixed',
              left: 0,
              top: `${animalConfig.y}vh`,
              width: `${animalConfig.size}px`,
              height: `${animalConfig.size}px`,
              objectFit: 'contain',
              pointerEvents: 'none',
              zIndex: 1000,
              filter: 'drop-shadow(0 4px 8px rgba(0, 0, 0, 0.2))',
            }}
          />
        )}
      </AnimatePresence>

      <div style={{ maxWidth: '600px', margin: '0 auto', padding: '20px' }}>
        <h3 style={{ color: '#000', textAlign: 'center', marginBottom: '20px', fontSize: '24px', fontWeight: '600' }}>Captcha 2</h3>
        <button
          onClick={handleBack}
          style={{
            marginBottom: '20px',
            padding: '10px 24px',
            fontSize: '16px',
            fontWeight: '600',
            color: '#667eea',
            background: 'white',
            border: '2px solid #667eea',
            borderRadius: '8px',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
          onMouseOver={(e) => {
            e.target.style.background = '#667eea';
            e.target.style.color = 'white';
          }}
          onMouseOut={(e) => {
            e.target.style.background = 'white';
            e.target.style.color = '#667eea';
          }}
        >
          ← Back
        </button>
        <DialRotationCaptcha
          onSuccess={() => navigate('/animal-selection', {
            state: { seenAnimal: animalConfig.animal.name }
          })}
        />
      </div>
    </div>
  );
}

export default AnimalRotationPage;
