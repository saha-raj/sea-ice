/**
 * Sea Ice Visualization - Controls
 * 
 * This file handles camera controls and user interface elements
 * for the sea ice visualization.
 */

// Initialize camera controls
function initCameraControls(camera, renderer, options = {}) {
  // Create orbit controls
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  
  // Set up control properties with defaults
  controls.enableDamping = options.enableDamping !== undefined ? options.enableDamping : true;
  controls.dampingFactor = options.dampingFactor || 0.05;
  controls.rotateSpeed = options.rotateSpeed || 0.5;
  controls.zoomSpeed = options.zoomSpeed || 1.0;
  controls.panSpeed = options.panSpeed || 0.8;
  
  // Set distance limits
  controls.minDistance = options.minDistance || 3;
  controls.maxDistance = options.maxDistance || 10;
  
  // Optional: Limit vertical rotation
  if (options.limitVerticalRotation) {
    controls.minPolarAngle = Math.PI * 0.1; // Limit top view
    controls.maxPolarAngle = Math.PI * 0.9; // Limit bottom view
  }
  
  // Disable pan if specified
  controls.enablePan = options.enablePan !== undefined ? options.enablePan : true;
  
  return controls;
}

// Create time controls for navigating through ice data
function createTimeControls(container, callback) {
  // Create controller elements
  const controlsDiv = document.createElement('div');
  controlsDiv.className = 'ice-time-controls';
  
  // Create time label
  const timeLabel = document.createElement('div');
  timeLabel.className = 'time-label';
  timeLabel.id = 'ice-data-label';
  timeLabel.textContent = 'Loading data...';
  
  // Create slider
  const slider = document.createElement('input');
  slider.type = 'range';
  slider.className = 'time-slider';
  slider.min = 0;
  slider.max = 100;
  slider.value = 0;
  slider.disabled = true;
  
  // Create play/pause button
  const playButton = document.createElement('button');
  playButton.className = 'play-button';
  playButton.innerHTML = '▶'; // Play icon
  playButton.disabled = true;
  
  // Add event listeners
  slider.addEventListener('input', function() {
    const value = parseInt(slider.value);
    if (callback && typeof callback.onSliderChange === 'function') {
      callback.onSliderChange(value);
    }
  });
  
  playButton.addEventListener('click', function() {
    const isPlaying = playButton.innerHTML === '❚❚'; // Pause icon
    
    // Toggle button state
    playButton.innerHTML = isPlaying ? '▶' : '❚❚';
    
    // Call the appropriate callback
    if (callback) {
      if (isPlaying && typeof callback.onPause === 'function') {
        callback.onPause();
      } else if (!isPlaying && typeof callback.onPlay === 'function') {
        callback.onPlay();
      }
    }
  });
  
  // Assemble controls
  controlsDiv.appendChild(timeLabel);
  controlsDiv.appendChild(slider);
  controlsDiv.appendChild(playButton);
  
  // Add to container
  container.appendChild(controlsDiv);
  
  // Return functions to update the controls
  return {
    updateTimeRange: function(min, max) {
      slider.min = min;
      slider.max = max;
      slider.disabled = false;
    },
    
    updateCurrentTime: function(value, label) {
      slider.value = value;
      timeLabel.textContent = label || `Data point ${value}`;
    },
    
    enablePlayButton: function() {
      playButton.disabled = false;
    },
    
    setPlaying: function(isPlaying) {
      playButton.innerHTML = isPlaying ? '❚❚' : '▶';
    }
  };
} 