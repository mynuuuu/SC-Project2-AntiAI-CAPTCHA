import { useRef, useState, useCallback } from 'react';

/**
 * Custom hook for tracking user behavior during captcha interaction
 * Captures mouse movements, clicks, and other events with detailed metrics
 */
const useBehaviorTracking = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const eventsRef = useRef([]);
  const startTimeRef = useRef(null);
  const lastEventTimeRef = useRef(null);
  const lastPositionRef = useRef({ x: 0, y: 0 });
  const statsRef = useRef({ moves: 0, clicks: 0, drags: 0 });

  // Generate unique session ID
  const generateSessionId = useCallback(() => {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);

  // Start recording behavior
  const startRecording = useCallback(() => {
    const newSessionId = generateSessionId();
    setSessionId(newSessionId);
    setIsRecording(true);
    startTimeRef.current = performance.now();
    lastEventTimeRef.current = startTimeRef.current;
    eventsRef.current = [];
    statsRef.current = { moves: 0, clicks: 0, drags: 0 };
    lastPositionRef.current = { x: 0, y: 0 };

    console.log('Behavior recording started:', newSessionId);
    return newSessionId;
  }, [generateSessionId]);

  // Stop recording behavior
  const stopRecording = useCallback(() => {
    setIsRecording(false);
    console.log('Behavior recording stopped. Total events:', eventsRef.current.length);
    return eventsRef.current;
  }, []);

  // Capture an event
  const captureEvent = useCallback((eventType, event, targetElement) => {
    if (!isRecording) return;

    const currentTime = performance.now();
    const timeSinceStart = currentTime - startTimeRef.current;
    const timeSinceLastEvent = lastEventTimeRef.current 
      ? currentTime - lastEventTimeRef.current 
      : 0;

    // Get coordinates relative to the target element if provided
    let relativeX = event.clientX;
    let relativeY = event.clientY;
    
    if (targetElement) {
      const rect = targetElement.getBoundingClientRect();
      relativeX = event.clientX - rect.left;
      relativeY = event.clientY - rect.top;
    }

    // Calculate velocity
    let velocity = 0;
    if (eventsRef.current.length > 0) {
      const lastEvent = eventsRef.current[eventsRef.current.length - 1];
      const distance = Math.sqrt(
        Math.pow(event.clientX - lastEvent.client_x, 2) +
        Math.pow(event.clientY - lastEvent.client_y, 2)
      );
      velocity = timeSinceLastEvent > 0 ? (distance / timeSinceLastEvent) * 1000 : 0;
    }

    // Calculate acceleration (change in velocity)
    let acceleration = 0;
    if (eventsRef.current.length > 0) {
      const lastEvent = eventsRef.current[eventsRef.current.length - 1];
      const velocityChange = velocity - (lastEvent.velocity || 0);
      acceleration = timeSinceLastEvent > 0 ? (velocityChange / timeSinceLastEvent) * 1000 : 0;
    }

    // Calculate direction (angle of movement)
    let direction = 0;
    if (eventsRef.current.length > 0) {
      const lastEvent = eventsRef.current[eventsRef.current.length - 1];
      const deltaX = event.clientX - lastEvent.client_x;
      const deltaY = event.clientY - lastEvent.client_y;
      direction = Math.atan2(deltaY, deltaX) * (180 / Math.PI);
    }

    const eventData = {
      session_id: sessionId,
      timestamp: Date.now(),
      time_since_start: timeSinceStart.toFixed(2),
      time_since_last_event: timeSinceLastEvent.toFixed(2),
      event_type: eventType,
      client_x: event.clientX,
      client_y: event.clientY,
      relative_x: relativeX.toFixed(2),
      relative_y: relativeY.toFixed(2),
      page_x: event.pageX || event.clientX,
      page_y: event.pageY || event.clientY,
      screen_x: event.screenX || 0,
      screen_y: event.screenY || 0,
      button: event.button !== undefined ? event.button : -1,
      buttons: event.buttons !== undefined ? event.buttons : 0,
      ctrl_key: event.ctrlKey || false,
      shift_key: event.shiftKey || false,
      alt_key: event.altKey || false,
      meta_key: event.metaKey || false,
      velocity: velocity.toFixed(2),
      acceleration: acceleration.toFixed(2),
      direction: direction.toFixed(2),
      user_agent: navigator.userAgent,
      screen_width: window.screen.width,
      screen_height: window.screen.height,
      viewport_width: window.innerWidth,
      viewport_height: window.innerHeight,
    };

    eventsRef.current.push(eventData);
    lastEventTimeRef.current = currentTime;
    lastPositionRef.current = { x: event.clientX, y: event.clientY };

    // Update stats
    if (eventType === 'mousemove') statsRef.current.moves++;
    if (eventType === 'click' || eventType === 'mousedown') statsRef.current.clicks++;
    if (eventType === 'drag') statsRef.current.drags++;

  }, [isRecording, sessionId]);

  // Convert events to CSV format
  const convertToCSV = useCallback((events) => {
    if (events.length === 0) return '';

    const headers = Object.keys(events[0]);
    const csvRows = [];

    // Add header row
    csvRows.push(headers.join(','));

    // Add data rows
    for (const row of events) {
      const values = headers.map(header => {
        const value = row[header];
        // Escape values that contain commas or quotes
        if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
          return `"${value.replace(/"/g, '""')}"`;
        }
        return value;
      });
      csvRows.push(values.join(','));
    }

    return csvRows.join('\n');
  }, []);

  // Download CSV file
  const downloadCSV = useCallback((csvContent, filename) => {
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');

    if (navigator.msSaveBlob) { // IE 10+
      navigator.msSaveBlob(blob, filename);
    } else {
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }

    console.log(`CSV saved as ${filename} with ${eventsRef.current.length} events`);
  }, []);

  // Save current session to CSV
  const saveToCSV = useCallback((filenamePrefix = 'behavior') => {
    if (eventsRef.current.length === 0) {
      console.warn('No events to save!');
      return false;
    }

    const csvContent = convertToCSV(eventsRef.current);
    const filename = `${filenamePrefix}_${sessionId}.csv`;
    downloadCSV(csvContent, filename);
    return true;
  }, [sessionId, convertToCSV, downloadCSV]);

  // Send data to server (for captcha-specific files)
  const sendToServer = useCallback(async (serverUrl, captchaType, userType = 'human', captchaId = 'captcha1', success = false) => {
    if (eventsRef.current.length === 0) {
      console.warn('No events to send!');
      return { success: false, error: 'No events' };
    }

    try {
      const response = await fetch(serverUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          captcha_id: captchaId,
          session_id: sessionId,
          captchaType: captchaType,
          events: eventsRef.current,
          metadata: {
            user_agent: navigator.userAgent,
            screen_width: window.screen.width,
            screen_height: window.screen.height,
            viewport_width: window.innerWidth,
            viewport_height: window.innerHeight,
          },
          success: success,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Data sent to server successfully:', result);
        return { success: true, result };
      } else {
        console.error('Server error:', response.statusText);
        return { success: false, error: response.statusText };
      }
    } catch (error) {
      console.error('Network error:', error);
      return { success: false, error: error.message };
    }
  }, [sessionId]);

  // Get current stats
  const getStats = useCallback(() => {
    return {
      eventCount: eventsRef.current.length,
      duration: startTimeRef.current 
        ? ((performance.now() - startTimeRef.current) / 1000).toFixed(2)
        : 0,
      moves: statsRef.current.moves,
      clicks: statsRef.current.clicks,
      drags: statsRef.current.drags,
    };
  }, []);

  // Reset tracking
  const reset = useCallback(() => {
    setIsRecording(false);
    setSessionId(null);
    eventsRef.current = [];
    startTimeRef.current = null;
    lastEventTimeRef.current = null;
    statsRef.current = { moves: 0, clicks: 0, drags: 0 };
    lastPositionRef.current = { x: 0, y: 0 };
  }, []);

  return {
    isRecording,
    sessionId,
    startRecording,
    stopRecording,
    captureEvent,
    saveToCSV,
    sendToServer,
    getStats,
    reset,
    events: eventsRef.current,
  };
};

export default useBehaviorTracking;

