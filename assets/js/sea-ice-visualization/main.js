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
  // Simulate fetching the list of files (normally this would be from an API)
  // For now we'll create data for July 2012 onwards
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
  for (let year = 2013; year <= 2024; year++) {
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
    if (tube && scene) { // Check if tube and scene exist
      scene.remove(tube);
      // --- ADD: Dispose geometry and material ---
      if (tube.geometry) tube.geometry.dispose();
      if (tube.material) tube.material.dispose();
      // --- END ADD ---
    }
  });
  
  // Clear the outlines object
  outlineTubes = {};
  hasResetOutlines = true;
  // console.log("NH: All outlines reset."); // Optional log
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
      // --- EDIT: Handle potential FeatureCollection ---
      // Check if the root is a FeatureCollection
      let coordinates;
      if (data.type === 'FeatureCollection' && data.features && data.features.length > 0) {
        // Use the coordinates from the first feature's geometry
        // Assuming the first feature is the primary outline we want
        if (data.features[0].geometry && data.features[0].geometry.type === 'LineString') {
          coordinates = data.features[0].geometry.coordinates;
        } else {
          console.warn(`NH: First feature in ${outlinePath} is not a LineString.`);
          return; // Skip if the first feature isn't a LineString
        }
      } else if (data.type === 'Feature' && data.geometry && data.geometry.type === 'LineString') {
        // Handle the old single Feature format
        coordinates = data.geometry.coordinates;
      } else {
        console.error(`NH: Invalid GeoJSON structure in ${outlinePath}:`, data);
        return; // Stop if the structure is not recognized
      }

      if (!coordinates || coordinates.length < 2) {
        console.warn(`NH: No valid coordinates found or not enough points in ${outlinePath}`);
        return;
      }
      // --- END EDIT ---


      // Create a line geometry from the coordinates
      const points = [];

      // Convert lon/lat to 3D positions
      coordinates.forEach(coord => {
        const lon = coord[0];
        const lat = coord[1];

        // Convert to radians
        const phi = (90 - lat) * (Math.PI / 180);
        const theta = (lon + 180) * (Math.PI / 180);

        // Calculate position on sphere with slight offset
        const radius = 2.002; // Simplified radius for NH
        const x = -radius * Math.sin(phi) * Math.cos(theta);
        const y = radius * Math.cos(phi);
        const z = radius * Math.sin(phi) * Math.sin(theta);

        points.push(new THREE.Vector3(x, y, z));
      });

      if (points.length < 2) {
          console.warn(`NH: Not enough valid points after conversion for ${outlinePath}`);
          return;
      }


      // Create a smooth curve from the points
      const curve = new THREE.CatmullRomCurve3(points, false); // Use false for LineString

      // Create a tube geometry around the curve
      const tubeGeometry = new THREE.TubeGeometry(
        curve,           // path
        Math.max(1, points.length * 2), // tubularSegments
        0.004,            // radius of tube
        8,               // radiusSegments (reduced for potential performance)
        false             // closed (false for LineString)
      );

      // --- EDIT: Make material transparent ---
      // Create material with the requested color
      const material = new THREE.MeshBasicMaterial({
        color: 0xe9c46a,   // Existing NH color
        transparent: true, // Enable transparency
        opacity: 1.0       // Start fully opaque
      });
      // --- END EDIT ---

      // Create the tube mesh and add it to the scene
      const tube = new THREE.Mesh(tubeGeometry, material);
      scene.add(tube);

      // Store the outline in our outlines object
      // If multiple tubes per year were needed, this would need changing
      outlineTubes[year] = tube;
    })
    .catch(error => console.error(`NH: Error loading or processing outline file ${outlinePath}:`, error));
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


  // --- FADING LOGIC START (Copied from SH) ---
  const FADE_RATE = 0.004; // How much opacity decreases each frame
  const REMOVE_THRESHOLD = 0.01; // Opacity level below which tubes are removed

  // Iterate through all years that have outlines
  Object.keys(outlineTubes).forEach(year => {
    const tube = outlineTubes[year]; // NH stores single tube per year currently

    // Check if it's a valid tube with material
    if (tube && tube.material) {
        // Decrease opacity
        tube.material.opacity -= FADE_RATE;

        // Check if tube should be removed based on opacity
        if (tube.material.opacity <= REMOVE_THRESHOLD) {
            // Opacity is too low, remove the tube now
            if (scene) scene.remove(tube);
            if (tube.geometry) tube.geometry.dispose();
            if (tube.material) tube.material.dispose();
            delete outlineTubes[year]; // Remove the entry for this year
            // console.log(`NH: Removed faded outline tube for year ${year}`); // Optional log
        }
    } else if (tube) {
        // If tube exists but material doesn't (shouldn't happen), clean up
        delete outlineTubes[year];
    }
  });
  // --- FADING LOGIC END ---


  if (renderer && scene && camera) {
    renderer.render(scene, camera);
  }
} 