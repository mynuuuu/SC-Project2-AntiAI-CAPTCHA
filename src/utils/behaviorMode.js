// #Author: Naman Taneja
const ATTACK_FLAG_KEY = 'attackModeActive';

const hasBrowserContext = () => typeof window !== 'undefined';

const setAttackFlag = () => {
  try {
    if (hasBrowserContext()) {
      window.sessionStorage.setItem(ATTACK_FLAG_KEY, 'true');
    }
  } catch (error) {
    console.warn('Unable to persist attack mode flag:', error);
  }
};

const readAttackFlag = () => {
  if (!hasBrowserContext()) {
    return false;
  }

  try {
    return window.sessionStorage.getItem(ATTACK_FLAG_KEY) === 'true';
  } catch (error) {
    console.warn('Unable to read attack mode flag:', error);
    return false;
  }
};

export const isAttackModeActive = () => {
  if (!hasBrowserContext()) {
    return false;
  }

  try {
    const params = new URLSearchParams(window.location.search || '');
    const attackParam = params.get('attackMode');

    if (attackParam === '1' || attackParam === 'true') {
      setAttackFlag();
      return true;
    }

    return readAttackFlag();
  } catch (error) {
    console.warn('Unable to determine attack mode:', error);
    return false;
  }
};

export const shouldCaptureBehavior = () => {
  return process.env.NODE_ENV === 'development' && !isAttackModeActive();
};

