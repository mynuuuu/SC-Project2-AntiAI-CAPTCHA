// finalHumanGate.js
// Centralized logic to decide what happens after the final (Morse) CAPTCHA.
// Uses the classification result stored from the slider sessions (captcha1/2/3)
// It decides:
// - 'allow'  → redirect to entry form (Next button enabled)
// - 'retry'  → give human another chance
// - 'block'  → redirect to bot detected page

export async function evaluateFinalAccess({ isSolved, sessionId }) {
  try {
    // Get the classification from the slider sessions (stored by AnimalSelectionPage)
    const storedClassification = localStorage.getItem('final_classification');
    
    if (!storedClassification) {
      // Fallback: try to get from slider classification
      const sliderClassification = localStorage.getItem('slider_classification');
      if (sliderClassification) {
        const classification = JSON.parse(sliderClassification);
        const isHuman = classification.is_human === true;
        
        if (!isHuman) {
          return 'block';
        }
        
        if (isSolved) {
          return 'allow';
        }
        
        return 'retry';
      }
      
      // No classification found - be conservative, allow retry
      console.warn('⚠️ No classification found in localStorage, defaulting to retry');
      return 'retry';
    }

    const classification = JSON.parse(storedClassification);
    const isHuman = classification.is_human === true;

    if (!isHuman) {
      // Classified as bot → straight block
      return 'block';
    }

    if (isSolved) {
      // Human and solved → allow entry form (Next button will be enabled)
      // Store flag that CAPTCHA passed
      localStorage.setItem('captcha_passed', 'true');
      localStorage.setItem('is_human', 'true');
      return 'allow';
    }

    // Human but did not solve Morse → give another chance
    return 'retry';
  } catch (e) {
    console.error('Error in evaluateFinalAccess:', e);
    // On error, be conservative: human gets another chance but no automatic allow.
    return 'retry';
  }
}


