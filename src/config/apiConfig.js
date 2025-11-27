/**
 * Centralized config for the behavior tracking backend.
 * Allows switching between dev (localhost) and prod (Render) via env vars.
 */
const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL || 'http://localhost:5001';

export const BEHAVIOR_API = {
  baseUrl: API_BASE_URL,
  saveEvents: `${API_BASE_URL}/save_events`,
  saveCaptchaEvents: `${API_BASE_URL}/save_captcha_events`,
  stats: `${API_BASE_URL}/stats`,
  sessions: `${API_BASE_URL}/sessions`,
};

export const withBehaviorEndpoint = (path) => `${API_BASE_URL}${path}`;

export default BEHAVIOR_API;

