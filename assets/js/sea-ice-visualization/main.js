/**
 * Sea Ice 3D Visualization - Main Entry Point
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Global variables
let scene, camera, renderer;
let earth;
let controls;

// Animation-related variables
let iceLayer = null;
let outlineTubes = {}; // Object to store outlines by year
let animationInterval = null;
let currentImageIndex = 0;
let isVisible = false;
let imagesData = []; // Will hold chronologically sorted dates in format {date: 'YYYY-MM-DD', imageFile: 'path', outlineFile: 'path'}
let yearStartIndices = {}; // To track where each year starts in the array
let hasResetOutlines = true; // Flag to track if we've reset outlines at the beginning of animation
let dateDisplay = null; // Element to display current date
let sourceDisplay = null; // Element to display the source
let debugDisplay = null; // TEMPORARY: Element to display camera position and target

// Initialize when the document is loaded
document.addEventListener('DOMContentLoaded', function() {
  initScene();
  createOverlayElements();
  loadDatesData();
  setupVisibilityObserver();
});

// Create overlay elements for displaying information
function createOverlayElements() {
  const container = document.getElementById('sea-ice-visualization');
  if (!container) return;
  
  // Create date display element (top left)
  dateDisplay = document.createElement('div');
  dateDisplay.className = 'sea-ice-date-display';
  container.appendChild(dateDisplay);
  
  // Create source display element (bottom left)
  sourceDisplay = document.createElement('div');
  sourceDisplay.className = 'sea-ice-source-display';
  sourceDisplay.innerHTML = 'Source: Institute of Environmental Physics, University of Bremen, Germany';
  container.appendChild(sourceDisplay);
  
  // TEMPORARY: Create debug display for camera position and target (bottom right)
  debugDisplay = document.createElement('div');
  debugDisplay.style.position = 'absolute';
  debugDisplay.style.bottom = '20px';
  debugDisplay.style.right = '20px';
  debugDisplay.style.fontFamily = 'monospace';
  debugDisplay.style.fontSize = '12px';
  debugDisplay.style.color = '#ffffff';
  debugDisplay.style.backgroundColor = 'rgba(0,0,0,0.5)';
  debugDisplay.style.padding = '10px';
  debugDisplay.style.borderRadius = '4px';
  debugDisplay.style.zIndex = '100';
  debugDisplay.style.textAlign = 'left';
  debugDisplay.style.whiteSpace = 'pre';
  container.appendChild(debugDisplay);
}

// Setup Intersection Observer to detect when visualization enters viewport
function setupVisibilityObserver() {
  const container = document.getElementById('sea-ice-visualization');
  if (!container) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        if (!isVisible) {
          isVisible = true;
          startAnimation();
        }
      } else {
        isVisible = false;
        stopAnimation();
      }
    });
  }, { threshold: 0.2 }); // Trigger when 20% of the element is visible

  observer.observe(container);
}

// Load and sort available dates
function loadDatesData() {
  // Pattern to match: yyyy-mm-15.png
  const datePattern = /(\d{4})-(\d{2})-15/;
  
  // Simulate fetching the list of files (normally this would be from an API)
  // For now we'll create data for 2012-2023, starting from July 2012
  const dates = [];
  
  // Add dates from July 2012 to December 2012 first
  for (let month = 7; month <= 12; month++) {
    const monthStr = month.toString().padStart(2, '0');
    const dateStr = `2012-${monthStr}-15`;
    dates.push({
      date: dateStr,
      year: '2012',
      month: monthStr,
      imageFile: `data/sea-ice-NH-images/${dateStr}.png`,
      outlineFile: `data/sea-ice-NH-outlines/${dateStr}.geojson`
    });
  }
  
  // Add full years from 2013 to 2024
  for (let year = 2012; year <= 2024; year++) {
    for (let month = 1; month <= 12; month++) {
      const monthStr = month.toString().padStart(2, '0');
      const dateStr = `${year}-${monthStr}-15`;
      dates.push({
        date: dateStr,
        year: year.toString(),
        month: monthStr,
        imageFile: `data/sea-ice-NH-images/${dateStr}.png`,
        outlineFile: `data/sea-ice-NH-outlines/${dateStr}.geojson`
      });
    }
  }
  
  // Sort chronologically
  imagesData = dates.sort((a, b) => a.date.localeCompare(b.date));
  
  // Track where each year starts in the sorted array
  imagesData.forEach((item, index) => {
    if (!yearStartIndices[item.year]) {
      yearStartIndices[item.year] = index;
    }
  });
  
  console.log(`Loaded ${imagesData.length} dates for animation`);
}

// Start animation cycle
function startAnimation() {
  if (animationInterval) return;
  
  // Reset outlines at the start of a new animation cycle
  if (hasResetOutlines === false) {
    resetOutlines();
  }
  
  // Load the first image immediately
  updateIceData(currentImageIndex);
  
  // Set interval for animation (change every 100 milliseconds)
  animationInterval = setInterval(() => {
    currentImageIndex = (currentImageIndex + 1) % imagesData.length;
    
    // If we've looped back to the beginning, reset the outlines
    if (currentImageIndex === 0) {
      resetOutlines();
    }
    
    updateIceData(currentImageIndex);
  }, 200);
}

// Stop animation cycle
function stopAnimation() {
  if (animationInterval) {
    clearInterval(animationInterval);
    animationInterval = null;
  }
}

// Reset all outlines
function resetOutlines() {
  // Remove all outline tubes
  Object.values(outlineTubes).forEach(tube => {
    if (tube) {
      scene.remove(tube);
    }
  });
  
  // Clear the outlines object
  outlineTubes = {};
  hasResetOutlines = true;
}

// Update ice data with new texture and outline
function updateIceData(index) {
  const data = imagesData[index];
  
  // Update ice texture
  updateIceTexture(data.imageFile);
  
  // Only show outline when month is September (09)
  if (data.month === '09') {
    // Only add this September outline if we don't already have it for this year
    if (!outlineTubes[data.year]) {
      updateIceOutline(data.outlineFile, data.year);
      hasResetOutlines = false;
    }
  }
  
  // Update the date display with current date
  updateDateDisplay(data);
}

// Update the date display with current month/year
function updateDateDisplay(data) {
  if (!dateDisplay) return;
  
  // Convert numeric month to name
  const monthNames = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
  ];
  
  // Get month index (0-11) from month string (01-12)
  const monthIndex = parseInt(data.month, 10) - 1;
  const monthName = monthNames[monthIndex];
  
  // Clear existing content
  dateDisplay.innerHTML = '';
  
  // Add year (bold)
  const yearSpan = document.createElement('span');
  yearSpan.textContent = data.year + ' ';
  dateDisplay.appendChild(yearSpan);
  
  // Add month (not bold)
  const monthSpan = document.createElement('span');
  monthSpan.style.fontWeight = 'normal';
  monthSpan.textContent = monthName;
  dateDisplay.appendChild(monthSpan);
}

// Update ice texture
function updateIceTexture(texturePath) {
  const textureLoader = new THREE.TextureLoader();
  textureLoader.load(texturePath, function(texture) {
    texture.minFilter = THREE.NearestFilter;
    texture.magFilter = THREE.NearestFilter;
    
    if (iceLayer) {
      // Update existing layer
      iceLayer.material.map = texture;
      iceLayer.material.needsUpdate = true;
    }
  }, undefined, function(err) {
    console.error('Error loading texture:', err);
  });
}

// Update ice outline
function updateIceOutline(outlinePath, year) {
  // Load new outline
  fetch(outlinePath)
    .then(response => response.json())
    .then(data => {
      const coordinates = data.geometry.coordinates;
      
      // Create a line geometry from the coordinates
      const points = [];
      
      // Convert lon/lat to 3D positions
      coordinates.forEach(coord => {
        const lon = coord[0];
        const lat = coord[1];
        
        // Convert to radians
        const phi = (90 - lat) * (Math.PI / 180);
        const theta = (lon + 180) * (Math.PI / 180);
        
        // Calculate position on sphere with slight offset to avoid z-fighting
        // Add a tiny bit more offset for each year to avoid z-fighting between outlines
        const yearOffset = (parseInt(year) - 2015) * 0.0001;
        const radius = 2.002 + yearOffset;
        const x = -radius * Math.sin(phi) * Math.cos(theta);
        const y = radius * Math.cos(phi);
        const z = radius * Math.sin(phi) * Math.sin(theta);
        
        points.push(new THREE.Vector3(x, y, z));
      });
      
      // Create a smooth curve from the points
      const curve = new THREE.CatmullRomCurve3(points, true); // true makes it a closed curve
      
      // Create a tube geometry around the curve (this gives actual thickness unlike linewidth)
      const tubeGeometry = new THREE.TubeGeometry(
        curve,           // path
        points.length * 2, // tubularSegments - more segments = smoother tube
        0.003,            // radius of tube
        20,               // radiusSegments - more segments = smoother tube cross-section
        true             // closed
      );
      
      // Create material with the requested color
      const material = new THREE.MeshBasicMaterial({ 
        // color: 0xeb5e28,
        color: 0x6AA6C8,
        transparent: true,
        opacity: 0.9
      });
      
      // Create the tube mesh and add it to the scene
      const tube = new THREE.Mesh(tubeGeometry, material);
      scene.add(tube);
      
      // Store the outline in our outlines object
      outlineTubes[year] = tube;
    })
    .catch(error => console.error('Error loading the outline file:', error));
}

// Initialize THREE.js scene
function initScene() {
  const container = document.getElementById('sea-ice-visualization');
  if (!container) {
    console.error('Container not found');
    return;
  }
  
  // Create scene
  scene = new THREE.Scene();
  
  // Create camera
  const aspect = container.clientWidth / container.clientHeight;
  camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
  // camera.position.set(0, 2.9, 1);  // Higher up and closer
  // camera.position.set(0, 2.77, 1.13);  // Higher up and closer
  camera.position.set(0, 3.25, 1.16);  // Higher up and closer  
  
  // Create renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  
  // Clear container and add renderer
  while (container.firstChild) {
    container.removeChild(container.firstChild);
  }
  container.appendChild(renderer.domElement);
  
  // Add lights
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambientLight);
  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(5, 3, 5);
  scene.add(directionalLight);
  
  // Create Earth
  const earthGeometry = new THREE.SphereGeometry(2, 64, 64);
  const earthMaterial = new THREE.MeshPhongMaterial({
    map: new THREE.TextureLoader().load('assets/images/2_no_clouds_8k_no_seaice.jpg'),
    shininess: 25
  });
  earth = new THREE.Mesh(earthGeometry, earthMaterial);
  scene.add(earth);
  
  // Create ice layer (initially without texture)
  const iceGeometry = new THREE.SphereGeometry(2, 64, 64);
  const iceMaterial = new THREE.MeshPhongMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.NormalBlending
  });
  iceLayer = new THREE.Mesh(iceGeometry, iceMaterial);
  scene.add(iceLayer);
  
  // Create controls
  controls = new OrbitControls(camera, renderer.domElement);
  // controls.target.set(0, 1.5, 0);  // Look slightly downward to shift Earth down in view
  // controls.target.set(0, 1.46, -0.02);  // Look slightly downward to shift Earth down in view
  controls.target.set(0, 1.41, 0.02);  // Look slightly downward to shift Earth down in view
  
  // Add window resize handler
  window.addEventListener('resize', onWindowResize);
  
  // Start animation loop
  animate();
}

// Handle window resize
function onWindowResize() {
  const container = document.getElementById('sea-ice-visualization');
  if (!container) return;
  
  const aspect = container.clientWidth / container.clientHeight;
  camera.aspect = aspect;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}

// Animation loop
function animate() {
  requestAnimationFrame(animate);
  if (controls) controls.update();
  if (renderer && scene && camera) {
    renderer.render(scene, camera);
    
    // TEMPORARY: Update debug display with camera position and target
    if (debugDisplay) {
      const pos = camera.position;
      const target = controls.target;
      debugDisplay.textContent = 
`Camera Position:
  x: ${pos.x.toFixed(2)}
  y: ${pos.y.toFixed(2)}
  z: ${pos.z.toFixed(2)}

Controls Target:
  x: ${target.x.toFixed(2)}
  y: ${target.y.toFixed(2)}
  z: ${target.z.toFixed(2)}`;
    }
  }
} 