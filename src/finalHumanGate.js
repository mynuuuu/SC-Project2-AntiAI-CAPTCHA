
export async function evaluateFinalAccess({ isSolved, sessionId }) {
  try {
    const classifications = [];
  
    const sliderClassification = localStorage.getItem('slider_classification');
    if (sliderClassification) {
      try {
        const classification = JSON.parse(sliderClassification);
        if (classification.prob_human !== undefined) {
          classifications.push({
            source: 'captcha1',
            prob_human: classification.prob_human,
            is_human: classification.is_human,
            decision: classification.decision
          });
          console.log(`Captcha1 classification: ${classification.decision} (prob: ${classification.prob_human.toFixed(3)})`);
        }
      } catch (e) {
        console.warn('Failed to parse slider_classification:', e);
      }
    }

    const rotationClassification = localStorage.getItem('rotation_classification');
    if (rotationClassification) {
      try {
        const classification = JSON.parse(rotationClassification);
        if (classification.prob_human !== undefined) {
          classifications.push({
            source: 'captcha2',
            prob_human: classification.prob_human,
            is_human: classification.is_human,
            decision: classification.decision
          });
          console.log(`Captcha2 classification: ${classification.decision} (prob: ${classification.prob_human.toFixed(3)})`);
        }
      } catch (e) {
        console.warn('Failed to parse rotation_classification:', e);
      }
    }

    const finalClassification = localStorage.getItem('final_classification');
    if (finalClassification) {
      try {
        const classification = JSON.parse(finalClassification);
        if (classification.prob_human !== undefined) {
          classifications.push({
            source: 'captcha3',
            prob_human: classification.prob_human,
            is_human: classification.is_human,
            decision: classification.decision
          });
          console.log(`Captcha3 classification: ${classification.decision} (prob: ${classification.prob_human.toFixed(3)})`);
        }
      } catch (e) {
        console.warn('Failed to parse final_classification:', e);
      }
    }
    
    // If no classifications found, be conservative and allow retry
    if (classifications.length === 0) {
      console.warn('No classification found in localStorage, defaulting to retry');
      return 'retry';
    }
    
    // Average the probabilities (ensemble method, similar to ML models)
    const avgProbHuman = classifications.reduce((sum, c) => sum + c.prob_human, 0) / classifications.length;
    const threshold = 0.7; // Same threshold as used in ML models
    const avgIsHuman = avgProbHuman >= threshold;
    
    console.log(`\n  Classification Averaging:`);
    console.log(`    Total classifications: ${classifications.length}`);
    console.log(`    Individual probabilities: ${classifications.map(c => c.prob_human.toFixed(3)).join(', ')}`);
    console.log(`    Averaged probability: ${avgProbHuman.toFixed(3)}`);
    console.log(`    Threshold: ${threshold}`);
    console.log(`    Final decision: ${avgIsHuman ? 'HUMAN' : 'BOT'}\n`);

    if (!avgIsHuman) {
      // Averaged classification indicates bot → straight block
      return 'block';
    }
    
    if (isSolved) {
      // Human (based on averaged classification) and solved Morse → allow entry form
      localStorage.setItem('captcha_passed', 'true');
      localStorage.setItem('is_human', 'true');
      return 'allow';
    }
    
    // Human (based on averaged classification) but did not solve Morse → give another chance
    return 'retry';
  } catch (e) {
    console.error('Error in evaluateFinalAccess:', e);
    // On error, be conservative: human gets another chance but no automatic allow.
    return 'retry';
  }
}


