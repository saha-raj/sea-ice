/**
 * Sea Ice Visualization - Utility Functions
 * 
 * This file contains utility functions used across the sea ice visualization.
 */

// Debounce function to limit how often a function can run
function debounce(func, wait) {
  let timeout;
  return function() {
    const context = this;
    const args = arguments;
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(context, args), wait);
  };
}

// Throttle function to limit execution rate
function throttle(func, limit) {
  let lastFunc;
  let lastRan;
  return function() {
    const context = this;
    const args = arguments;
    if (!lastRan) {
      func.apply(context, args);
      lastRan = Date.now();
    } else {
      clearTimeout(lastFunc);
      lastFunc = setTimeout(function() {
        if ((Date.now() - lastRan) >= limit) {
          func.apply(context, args);
          lastRan = Date.now();
        }
      }, limit - (Date.now() - lastRan));
    }
  };
}

// Linear interpolation between two values
function lerp(start, end, t) {
  return start * (1 - t) + end * t;
}

// Get the month name from a month number (0-11)
function getMonthName(month) {
  const monthNames = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];
  return monthNames[month];
}

// Parse the date from a filename in the format YYYY-MM.json
function parseDateFromFilename(filename) {
  const match = filename.match(/(\d{4})-(\d{2})\.json/);
  if (match) {
    const year = parseInt(match[1]);
    const month = parseInt(match[2]) - 1; // Convert to 0-11 format
    return {
      year: year,
      month: month,
      label: `${getMonthName(month)} ${year}`
    };
  }
  return null;
}

// Convert degrees to radians
function degToRad(degrees) {
  return degrees * (Math.PI / 180);
}

// Check if the WebGL is available and supported
function isWebGLAvailable() {
  try {
    const canvas = document.createElement('canvas');
    return !!(
      window.WebGLRenderingContext && 
      (canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
    );
  } catch (e) {
    return false;
  }
}

// Display a error message in the container if WebGL is not supported
function showWebGLError(container) {
  const errorDiv = document.createElement('div');
  errorDiv.className = 'webgl-error';
  errorDiv.innerHTML = `
    <p>
      Your browser does not support WebGL, which is required for this visualization.
      Please try using a modern browser like Chrome, Firefox, or Edge.
    </p>
  `;
  
  // Clear the container and add the error message
  while (container.firstChild) {
    container.removeChild(container.firstChild);
  }
  container.appendChild(errorDiv);
} 