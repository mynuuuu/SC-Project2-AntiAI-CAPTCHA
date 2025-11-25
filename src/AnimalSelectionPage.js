import React, { useRef, useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { animalImages } from './AnimalRotationPage';
import useBehaviorTracking from './useBehaviorTracking';
import { shouldCaptureBehavior } from './utils/behaviorMode';
import './App.css';

const CAPTCHA_ID = 'captcha3';

function AnimalSelectionPage() {
    const location = useLocation();
    const navigate = useNavigate();
    const { seenAnimal } = location.state || {};
    const [isSuccess, setIsSuccess] = useState(false);
    const [isAnswered, setIsAnswered] = useState(false);
    const containerRef = useRef(null);
    const mountTimeRef = useRef(null);
    const optionHoverTimesRef = useRef({});
    const optionHoverCountsRef = useRef({});
    const optionHoverStartTimesRef = useRef({});
    const mousePathRef = useRef([]);
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
        if (shouldLogBehavior) {
            startRecording();
            console.log('Layer 3: Started recording behavior');
        } else {
            console.log('Layer 3: Behavior logging is disabled');
        }
        
        // Track mouse movements for path analysis AND capture as events
        // Throttle to avoid too many events (capture every ~50ms for reasonable data size)
        let lastMouseMoveTime = 0;
        const throttledHandleMouseMove = (e) => {
            const now = performance.now();
            
            // Always track in mousePathRef for detailed path analysis
            if (shouldLogBehavior && !isAnswered) {
                mousePathRef.current.push({
                    x: e.clientX,
                    y: e.clientY,
                    t: performance.now() - mountTimeRef.current,
                });
                if (mousePathRef.current.length > 500) {
                    mousePathRef.current.shift();
                }
            }
            
            // Capture as event for behavior tracking (throttled to ~20 events/sec)
            if (shouldLogBehavior && !isAnswered && isRecording && (now - lastMouseMoveTime >= 50)) {
                captureEvent('mousemove', e, containerRef.current);
                lastMouseMoveTime = now;
            }
        };

        document.addEventListener('mousemove', throttledHandleMouseMove);
        return () => {
            document.removeEventListener('mousemove', throttledHandleMouseMove);
        };
    }, [shouldLogBehavior, isAnswered, isRecording, startRecording, captureEvent]);

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

    const handleSelect = async (animalName, optionIndex) => {
        const now = performance.now();
        const timeToAnswer = mountTimeRef.current ? now - mountTimeRef.current : 0;
        const isCorrect = animalName === seenAnimal;

        // Stop recording and send data
        const events = shouldLogBehavior ? stopRecording() : [];
        const behaviorStats = shouldLogBehavior ? getStats() : {};

        // Send data to server
        if (shouldLogBehavior) {
            // Calculate mouse path metrics
            const path = mousePathRef.current;
            let totalPathLength = 0;
            let pathStraightness = 0;
            
            if (path.length > 1) {
                for (let i = 1; i < path.length; i++) {
                    const dx = path[i].x - path[i-1].x;
                    const dy = path[i].y - path[i-1].y;
                    totalPathLength += Math.sqrt(dx * dx + dy * dy);
                }
                
                // Straightness: direct distance / actual path length
                if (path.length > 0 && totalPathLength > 0) {
                    const directDist = Math.sqrt(
                        Math.pow(path[path.length - 1].x - path[0].x, 2) +
                        Math.pow(path[path.length - 1].y - path[0].y, 2)
                    );
                    pathStraightness = directDist / totalPathLength;
                }
            }

            const metadata = {
                correct_animal: seenAnimal,
                selected_animal: animalName,
                success: isCorrect,
                option_index: optionIndex,
                time_to_answer_ms: Number(timeToAnswer.toFixed(2)),
                option_hover_times: optionHoverTimesRef.current,
                option_hover_counts: optionHoverCountsRef.current,
                total_hover_time_ms: Object.values(optionHoverTimesRef.current).reduce((sum, t) => sum + (typeof t === 'number' ? t : 0), 0),
                total_hover_count: Object.values(optionHoverCountsRef.current).reduce((sum, c) => sum + (typeof c === 'number' ? c : 0), 0),
                mouse_path_length_px: Number(totalPathLength.toFixed(2)),
                mouse_path_straightness: Number(pathStraightness.toFixed(4)),
                mouse_path_points: path.length,
                mouse_path: path.slice(-100),
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
                        captchaType: 'question',
                        events: events,
                        metadata: metadata,
                        success: isCorrect,
                    }),
                });

                const result = await response.json();
                
                if (result.success) {
                    console.log(`✓ Layer 3 data saved to ${CAPTCHA_ID}.csv`);
                } else {
                    console.error('Failed to save data:', result.error);
                }
            } catch (error) {
                console.error('Error saving data:', error);
            }
        } else {
            console.log('Layer 3: Behavior logging is disabled');
        }

        // Normal captcha verification flow
        setIsSuccess(isCorrect);
        setIsAnswered(true);
    };

    const handleNext = () => {
        navigate('/morse-captcha');
    }
    const handleOptionHover = (optionIndex, isEntering) => {
        if (!shouldLogBehavior || isAnswered) return;
        
        const now = performance.now();
        if (isEntering) {
            // Store start time for this hover
            optionHoverStartTimesRef.current[optionIndex] = now;
            optionHoverCountsRef.current[optionIndex] = (optionHoverCountsRef.current[optionIndex] || 0) + 1;
        } else {
            // Calculate time spent hovering and accumulate
            if (optionHoverStartTimesRef.current[optionIndex] !== undefined) {
                const hoverStart = optionHoverStartTimesRef.current[optionIndex];
                const hoverDuration = now - hoverStart;
                optionHoverTimesRef.current[optionIndex] = (optionHoverTimesRef.current[optionIndex] || 0) + hoverDuration;
                delete optionHoverStartTimesRef.current[optionIndex];
            }
        }
    };

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
        <div className="App" ref={containerRef}>
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
                        onClick={(e) => {
                            if (shouldLogBehavior) {
                                captureEvent('click', e.nativeEvent, containerRef.current);
                            }
                            handleSelect(animal.name, index);
                        }}
                        onMouseEnter={() => handleOptionHover(index, true)}
                        onMouseLeave={() => handleOptionHover(index, false)}
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
