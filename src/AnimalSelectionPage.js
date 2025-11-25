import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { animalImages } from './AnimalRotationPage';
import './App.css';

function AnimalSelectionPage() {
    const location = useLocation();
    const navigate = useNavigate();
    const { seenAnimal } = location.state || {};
    const [isSuccess, setIsSuccess] = React.useState(false);
    const [isAnswered, setIsAnswered] = React.useState(false);

    const [displayOptions] = React.useState(() => {
        if (!seenAnimal) return [];

        // Find the correct animal object
        const correctAnimal = animalImages.find(a => a.name === seenAnimal);

        // Filter out the correct animal to get distractors
        const distractors = animalImages.filter(a => a.name !== seenAnimal);

        // Shuffle distractors and pick 2
        const shuffledDistractors = distractors.sort(() => 0.5 - Math.random()).slice(0, 2);

        // Combine correct animal with distractors
        const options = [correctAnimal, ...shuffledDistractors];

        // Shuffle the final options
        return options.sort(() => 0.5 - Math.random());
    });

    const handleSelect = (animalName) => {
        if (isAnswered) return;
        if (animalName === seenAnimal) {
            setIsSuccess(true);
        } else {
            setIsSuccess(false);
        }
        setIsAnswered(true);
    };

    const handleNext = () => {
        navigate('/morse-captcha');
    }

    const handleFinish = () => {
        navigate('/');
    };

    const goHome = () => {
        navigate('/');
    }

    // if (isAnswered) {
    //     return (
    //         <div className="App">
    //             <header className="App-header">
    //                 <h1>Verification Complete</h1>
    //             </header>
    //             <div className="success-message" style={{ marginTop: '50px' }}>
    //                 <h2>{isSuccess ? '🎉 You are Human!' : '🚫 You are a Robot!'}</h2>
    //                 <p>{isSuccess ? 'You have successfully identified the animal.' : ''}</p>
    //                 <button
    //                     onClick={handleFinish}
    //                     style={{
    //                         marginTop: '20px',
    //                         padding: '12px 32px',
    //                         fontSize: '18px',
    //                         fontWeight: '600',
    //                         color: 'white',
    //                         background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    //                         border: 'none',
    //                         borderRadius: '25px',
    //                         cursor: 'pointer',
    //                         boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
    //                         transition: 'all 0.2s ease',
    //                     }}
    //                     onMouseOver={(e) => e.target.style.transform = 'translateY(-2px)'}
    //                     onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
    //                 >
    //                     Finish
    //                 </button>
    //             </div>
    //         </div>
    //     );
    // }

    return (
        <div className="App">
            <header className="App-header">
                <h1>Verification</h1>
                <p>Which floating animal did you see?</p>
            </header>
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '30px',
                padding: '40px',
                maxWidth: '1000px',
                margin: '0 auto'
            }}>
                {displayOptions.map((animal, index) => (
                    <div
                        key={index}
                        onClick={() => handleSelect(animal.name)}
                        style={{
                            cursor: 'pointer',
                            padding: '20px',
                            background: 'white',
                            borderRadius: '15px',
                            boxShadow: '0 10px 20px rgba(0,0,0,0.1)',
                            transition: 'all 0.3s ease',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            gap: '15px'
                        }}
                        onMouseOver={(e) => {
                            e.currentTarget.style.transform = 'translateY(-5px)';
                            e.currentTarget.style.boxShadow = '0 15px 30px rgba(0,0,0,0.2)';
                        }}
                        onMouseOut={(e) => {
                            e.currentTarget.style.transform = 'translateY(0)';
                            e.currentTarget.style.boxShadow = '0 10px 20px rgba(0,0,0,0.1)';
                        }}
                    >
                        <p style={{
                            margin: 0,
                            fontSize: '1.5rem',
                            fontWeight: '600',
                            color: '#2c3e50'
                        }}>{animal.name}</p>

                    </div>
                ))}
            </div>
            {isSuccess && (
                <div className="success-message">
                    <h2>Captcha Verified!</h2>
                    <p>Click Next to continue to the rotation captcha.</p>
                    <button
                        onClick={handleNext}
                        style={{
                            marginTop: '20px',
                            padding: '12px 32px',
                            fontSize: '18px',
                            fontWeight: '600',
                            color: 'white',
                            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                            border: 'none',
                            borderRadius: '25px',
                            cursor: 'pointer',
                            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
                            transition: 'all 0.2s ease',
                        }}
                        onMouseOver={(e) => e.target.style.transform = 'translateY(-2px)'}
                        onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
                    >
                        Next →
                    </button>
                </div>
            )}
            {!isSuccess && isAnswered && (
                <div className="success-message">
                    <h2>Captcha Verification Failed!</h2>
                    <button
                        onClick={goHome}
                        style={{
                            marginTop: '20px',
                            padding: '12px 32px',
                            fontSize: '18px',
                            fontWeight: '600',
                            color: 'white',
                            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                            border: 'none',
                            borderRadius: '25px',
                            cursor: 'pointer',
                            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
                            transition: 'all 0.2s ease',
                        }}
                        onMouseOver={(e) => e.target.style.transform = 'translateY(-2px)'}
                        onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
                    >
                        Start Over
                    </button>
                </div>
            )}
        </div>
    );
}

export default AnimalSelectionPage;
