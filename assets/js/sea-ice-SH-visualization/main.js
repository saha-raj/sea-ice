/**
 * Sea Ice 3D Visualization - Southern Hemisphere - Main Entry Point
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Global variables (prefixed with SH_)
let SH_scene, SH_camera, SH_renderer;
let SH_earth;
let SH_controls;

// Animation-related variables (prefixed with SH_)
let SH_iceLayer = null;
let SH_outlineTubes = {}; // Object to store outlines by year. Value will now be an ARRAY of tubes.
let SH_animationInterval = null;
let SH_currentImageIndex = 0;
let SH_isVisible = false;
let SH_imagesData = []; // Will hold chronologically sorted dates
let SH_yearStartIndices = {}; // To track where each year starts in the array
let SH_dateDisplay = null; // Element to display current date
let SH_sourceDisplay = null; // Element to display the source
let SH_debugDisplay = null; // TEMPORARY: Element to display camera position and target
let SH_hasResetOutlines = false; // Added to track if outlines have been reset

// Function to convert Lat/Lon to 3D point on sphere (prefixed function)
function SH_latLonToVector3(lat, lon, radius) {
  const phi = (90 - lat) * (Math.PI / 180); // Convert latitude to radians from the pole
  const theta = (lon + 180) * (Math.PI / 180); // Convert longitude to radians

  // Calculate x, y, z coordinates
  const x = -(radius * Math.sin(phi) * Math.cos(theta));
  const y = radius * Math.cos(phi); // Y is up/down in THREE.js
  const z = radius * Math.sin(phi) * Math.sin(theta);

  return new THREE.Vector3(x, y, z);
}

// Function to create a single outline tube from coordinates (prefixed function)
function SH_createOutlineTube(coordinates, radius = 2.02, tubeRadius = 0.005, color = 0xffa500) {
  if (!coordinates || coordinates.length < 2) {
    console.warn("SH: Not enough coordinates to create an outline tube.");
    return null;
  }

  // Convert GeoJSON [lon, lat] pairs to THREE.Vector3 points on the sphere
  const points = coordinates.map(coord => {
    // Ensure coord is an array [lon, lat]
    if (!Array.isArray(coord) || coord.length < 2) {
      console.warn("SH: Invalid coordinate pair found:", coord);
      return null; // Skip invalid points
    }
    const lon = coord[0];
    const lat = coord[1];
    // Add a check for valid numbers, though JSON parsing should handle this
    if (typeof lat !== 'number' || typeof lon !== 'number' || isNaN(lat) || isNaN(lon)) {
        console.warn(`SH: Invalid lat/lon values (${lat}, ${lon})`);
        return null; // Skip invalid points
    }
    return SH_latLonToVector3(lat, lon, radius);
  }).filter(p => p !== null); // Remove any nulls from invalid coords

  if (points.length < 2) {
      console.warn("SH: Not enough valid coordinates after conversion to create an outline tube.");
      return null;
  }


  // Create a curve from the points
  const curve = new THREE.CatmullRomCurve3(points, false); // false = not closed loop (LineString)

  // Create the tube geometry
  // Adjust tubularSegments and radialSegments for performance vs quality
  const tubeGeometry = new THREE.TubeGeometry(curve, Math.max(1, points.length * 2), tubeRadius, 8, false);

  // Create the material (adjust color, opacity etc. as needed)
  const tubeMaterial = new THREE.MeshBasicMaterial({
      color: color,
      transparent: true, // Enable transparency
      opacity: 1.0       // Start fully opaque
      // wireframe: true // Optional: for debugging
   });

  // Create the mesh
  const tubeMesh = new THREE.Mesh(tubeGeometry, tubeMaterial);
  return tubeMesh;
}

// Initialize when the document is loaded
document.addEventListener('DOMContentLoaded', function() {
  // Call prefixed functions
  SH_initScene();
  // Create overlays AFTER scene init, otherwise they get cleared
  SH_createOverlayElements();
  SH_loadDatesData(); // Make sure paths here are correct for SH data
  SH_setupVisibilityObserver();
});

// Create overlay elements for displaying information (prefixed function)
function SH_createOverlayElements() {
  const container = document.getElementById('sea-ice-SH-visualization');
  if (!container) return;

  // Create date display element (top left)
  SH_dateDisplay = document.createElement('div');
  SH_dateDisplay.className = 'sea-ice-date-display'; // Use same class as NH
  container.appendChild(SH_dateDisplay);

  // Create source display element (bottom left)
  SH_sourceDisplay = document.createElement('div');
  SH_sourceDisplay.className = 'sea-ice-source-display'; // Use same class as NH
  SH_sourceDisplay.innerHTML = 'Source: Institute of Environmental Physics, University of Bremen, Germany';
  container.appendChild(SH_sourceDisplay);

  // --- REMOVE Debug Display Creation ---
  /*
  // TEMPORARY: Create debug display for camera position and target (bottom right)
  SH_debugDisplay = document.createElement('div');
  SH_debugDisplay.style.position = 'absolute';
  SH_debugDisplay.style.bottom = '20px';
  SH_debugDisplay.style.right = '20px';
  SH_debugDisplay.style.fontFamily = 'monospace';
  SH_debugDisplay.style.fontSize = '12px';
  SH_debugDisplay.style.color = '#ffffff';
  SH_debugDisplay.style.backgroundColor = 'rgba(0,0,0,0.5)';
  SH_debugDisplay.style.padding = '10px';
  SH_debugDisplay.style.borderRadius = '4px';
  SH_debugDisplay.style.zIndex = '100';
  SH_debugDisplay.style.textAlign = 'left';
  SH_debugDisplay.style.whiteSpace = 'pre';
  container.appendChild(SH_debugDisplay);
  */
  // --- END REMOVE ---
}

// Setup Intersection Observer (prefixed function)
function SH_setupVisibilityObserver() {
  // Target the correct container ID
  const container = document.getElementById('sea-ice-SH-visualization');
  if (!container) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        if (!SH_isVisible) {
          SH_isVisible = true;
          SH_startAnimation(); // Call prefixed function
        }
      } else {
        SH_isVisible = false;
        SH_stopAnimation(); // Call prefixed function
      }
    });
  }, { threshold: 0.2 }); // Trigger when 20% of the element is visible

  observer.observe(container);
}

// Load and sort available dates (prefixed function)
// !!! IMPORTANT: This function needs modification to load SH data paths !!!
function SH_loadDatesData() {
  // Simulate fetching the list of files (normally this would be from an API)
  // For now we'll create data for July 2012 onwards
  // !!! THESE PATHS ARE STILL FOR NH - MUST BE CHANGED FOR SH !!!
  const dates = [];

  // Add dates from July 2012 to December 2012 first
  for (let month = 7; month <= 12; month++) {
    const monthStr = month.toString().padStart(2, '0');
    const dateStr = `2012-${monthStr}-15`;
    dates.push({
      date: dateStr,
      year: '2012',
      month: monthStr,
      // !!! CHANGE PATH !!!
      imageFile: `data/sea-ice-SH-images/${dateStr}.png`,
      // !!! CHANGE PATH !!!
      outlineFile: `data/sea-ice-SH-outlines/${dateStr}.geojson`
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
        // !!! CHANGE PATH !!!
        imageFile: `data/sea-ice-SH-images/${dateStr}.png`,
        // !!! CHANGE PATH !!!
        outlineFile: `data/sea-ice-SH-outlines/${dateStr}.geojson`
      });
    }
  }

  // Sort chronologically
  SH_imagesData = dates.sort((a, b) => a.date.localeCompare(b.date));

  // Track where each year starts in the sorted array
  SH_imagesData.forEach((item, index) => {
    if (!SH_yearStartIndices[item.year]) {
      SH_yearStartIndices[item.year] = index;
    }
  });

  console.log(`SH: Loaded ${SH_imagesData.length} dates for animation`); // Added SH prefix for logging
}

// Start animation cycle (prefixed function)
function SH_startAnimation() {
  if (SH_animationInterval) return; // Already running

  // --- ADD: Reset outlines at the start if needed (like NH) ---
  // Reset outlines at the start of a new animation cycle if they weren't already reset
  if (SH_hasResetOutlines === false) {
     SH_resetOutlines(); // Call prefixed function
  }
  // --- END ADD ---

  // Load the first image immediately
  SH_updateIceData(SH_currentImageIndex); // Call prefixed function

  // Set interval for animation (adjust timing as needed)
  SH_animationInterval = setInterval(() => {
    // Increment index and loop around
    SH_currentImageIndex = (SH_currentImageIndex + 1) % SH_imagesData.length;

    // --- ADD: Reset outlines on loop (like NH) ---
    // If we've looped back to the beginning, reset the outlines
    if (SH_currentImageIndex === 0) {
       SH_resetOutlines(); // Call prefixed function
    }
    // --- END ADD ---

    // Update texture and potentially outline for the new index
    SH_updateIceData(SH_currentImageIndex); // Call prefixed function
  }, 200); // Interval time (e.g., 200ms)
  console.log("SH: Animation interval started.");
}

// Stop animation cycle (prefixed function)
function SH_stopAnimation() {
  if (SH_animationInterval) {
    clearInterval(SH_animationInterval);
    SH_animationInterval = null;
  }
}

// Update ice data with new texture and outline (prefixed function)
function SH_updateIceData(index) {
  const data = SH_imagesData[index];
  if (!data) {
      console.error(`SH: No data found for index ${index}`);
      return;
  }

  // Update ice texture
  SH_updateIceTexture(data.imageFile); // Call prefixed function

  // Only show outline when month is March ('03')
  if (data.month === '02') {
    // Only add this outline if we don't already have it for this year
    // Note: SH_updateIceOutline already handles removing old tubes for THIS specific year
    // before adding new ones. We don't need an extra check here.
    SH_updateIceOutline(data.year, data.outlineFile); // Call prefixed function
  }

  // Update the date display with current date
  SH_updateDateDisplay(data); // Call prefixed function
}

// Update the date display with current month/year (prefixed function)
function SH_updateDateDisplay(data) {
  if (!SH_dateDisplay) return;

  // Convert numeric month to name
  const monthNames = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
  ];

  // Get month index (0-11) from month string (01-12)
  const monthIndex = parseInt(data.month, 10) - 1;
  const monthName = monthNames[monthIndex];

  // Clear existing content
  SH_dateDisplay.innerHTML = '';

  // Add year (bold)
  const yearSpan = document.createElement('span');
  yearSpan.textContent = data.year + ' ';
  SH_dateDisplay.appendChild(yearSpan);

  // Add month (not bold)
  const monthSpan = document.createElement('span');
  monthSpan.style.fontWeight = 'normal';
  monthSpan.textContent = monthName;
  SH_dateDisplay.appendChild(monthSpan);
}

// Update ice texture (prefixed function)
function SH_updateIceTexture(texturePath) {
  const textureLoader = new THREE.TextureLoader();
  textureLoader.load(texturePath, function(texture) {
    texture.minFilter = THREE.NearestFilter;
    texture.magFilter = THREE.NearestFilter;

    if (SH_iceLayer) { // Use prefixed variable
      // Update existing layer
      SH_iceLayer.material.map = texture;
      SH_iceLayer.material.needsUpdate = true;
    }
  }, undefined, function(err) {
    console.error('SH: Error loading texture:', texturePath, err); // Added SH prefix
  });
}

// Update ice outline based on GeoJSON data (prefixed function)
function SH_updateIceOutline(year, outlineFile) {
  // --- Existing cleanup logic for the specific year ---
  // First, remove any existing tubes specifically for THIS year before adding new ones
  if (SH_outlineTubes[year] && Array.isArray(SH_outlineTubes[year])) {
    // console.log(`SH: Removing ${SH_outlineTubes[year].length} existing outline tube(s) for year ${year} before update.`); // Optional log
    SH_outlineTubes[year].forEach(tube => {
      if (tube && SH_scene) {
        SH_scene.remove(tube);
        if (tube.geometry) tube.geometry.dispose();
        if (tube.material) tube.material.dispose();
      }
    });
  }
  // We will initialize SH_outlineTubes[year] below after fetch confirms data exists
  // --- End Existing cleanup ---


  fetch(outlineFile)
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status} for ${outlineFile}`);
      }
      return response.json();
    })
    .then(data => {
      // Validate the structure
      if (!data || data.type !== 'FeatureCollection' || !Array.isArray(data.features)) {
        console.error(`SH: Invalid GeoJSON structure or missing features in ${outlineFile}:`, data);
        SH_outlineTubes[year] = []; // Ensure it's an empty array even on error to prevent later issues
        return; // Stop processing this file
      }

      console.log(`SH: Processing ${data.features.length} features from ${outlineFile}`);

      // --- ADD: Initialize the array for this year ---
      // Ensure the array for this year exists before pushing tubes into it
      SH_outlineTubes[year] = [];
      // --- END ADD ---

      // Process each feature (outline) in the GeoJSON file
      data.features.forEach((feature, index) => {
        // Check if the feature has geometry and coordinates
        if (!feature || !feature.geometry || !feature.geometry.coordinates || feature.geometry.type !== 'LineString') {
          console.warn(`SH: Skipping feature ${index} in ${outlineFile} due to missing/invalid geometry or not being a LineString.`);
          return; // Skip this feature
        }

        const coordinates = feature.geometry.coordinates;
        // Define color and thickness for the outline
        const newThickness = 0.004; // Example thickness
        const newColor = 0xe9c46a; // Example color (yellow)

        // Create the tube mesh for this specific feature
        const tubeMesh = SH_createOutlineTube(coordinates, 2.02, newThickness, newColor); // Use prefixed function

        // If tube creation was successful, add it to the scene and the array for this year
        if (tubeMesh) {
          SH_scene.add(tubeMesh);
          // Store the tube mesh in the array for this year
          SH_outlineTubes[year].push(tubeMesh); // This should now work
        } else {
             console.warn(`SH: Failed to create tube for feature ${index} in ${outlineFile}`);
        }
      });
      console.log(`SH: Added ${SH_outlineTubes[year].length} outline tube(s) for ${year}`);

    })
    .catch(error => {
      console.error(`SH: Error loading or processing the outline file: ${outlineFile}`, error);
      // Ensure the entry for the year is at least an empty array to prevent downstream errors
      if (!SH_outlineTubes[year]) {
          SH_outlineTubes[year] = [];
      }
    });
}

// Initialize THREE.js scene (prefixed function) - RESTORING ELEMENTS
function SH_initScene() {
  console.log("SH: Initializing scene (Restoring elements)...");
  const container = document.getElementById('sea-ice-SH-visualization');
  if (!container) {
    console.error('SH: Container #sea-ice-SH-visualization not found');
    return;
  }

  // --- Basic Scene Setup ---
  try {
    SH_scene = new THREE.Scene();
    // No background color needed if Earth fills view
    // SH_scene.background = new THREE.Color(0x000000); // Optional: Black background
    console.log("SH: Scene created.");

    const width = container.clientWidth || 1;
    const height = container.clientHeight || 1;
    const aspect = width / height;
    SH_camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
    console.log(`SH: Camera created (aspect ${aspect.toFixed(2)}, size ${width}x${height})`);

    // !!! POSITION CAMERA FOR SH VIEW (e.g., below South Pole looking up) !!!
    SH_camera.position.set(0, -4.38, 0.03); // Position below the pole, slightly out
    console.log("SH: Camera position set for SH view.");

    SH_renderer = new THREE.WebGLRenderer({ antialias: true });
    SH_renderer.setPixelRatio(window.devicePixelRatio);
    SH_renderer.setSize(width, height);
    // --- REMOVE CLEAR COLOR ---
    // SH_renderer.setClearColor(0xff0000, 1); // Remove red clear color
    console.log(`SH: Renderer created (size ${width}x${height}).`);

    // --- Clear Container and Add Renderer ---
    console.log("SH: Clearing container...");
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }
    console.log("SH: Appending renderer DOM element...");
    container.appendChild(SH_renderer.domElement);
    if (container.querySelector('canvas')) {
        console.log("SH: Canvas element successfully appended.");
        const canvas = container.querySelector('canvas');
        console.log(`SH: Appended canvas size: ${canvas.width}x${canvas.height}`);
    } else {
        console.error("SH: Failed to append canvas element!");
        return;
    }

    // --- REMOVE SIMPLE CUBE ---
    /*
    const cubeGeometry = new THREE.BoxGeometry(1, 1, 1);
    const cubeMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    const cube = new THREE.Mesh(cubeGeometry, cubeMaterial);
    SH_scene.add(cube);
    console.log("SH: Removed simple green cube.");
    */

    // --- UNCOMMENT SCENE OBJECTS ---

    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5); // Soft white light
    SH_scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8); // Brighter directional light
    // Position light to illuminate Antarctica from below/side
    directionalLight.position.set(5, -5, 2);
    SH_scene.add(directionalLight);
    console.log("SH: Lights added.");

    // Create Earth (prefixed variable)
    // Ensure the texture path is correct
    const earthTexture = new THREE.TextureLoader().load('assets/images/2_no_clouds_8k_no_seaice.jpg',
        () => console.log("SH: Earth texture loaded successfully."), // Success callback
        undefined, // Progress callback (optional)
        (err) => console.error("SH: Error loading Earth texture:", err) // Error callback
    );
    const earthGeometry = new THREE.SphereGeometry(2, 64, 64); // Radius 2
    const earthMaterial = new THREE.MeshPhongMaterial({ map: earthTexture, shininess: 15 }); // Less shiny
    SH_earth = new THREE.Mesh(earthGeometry, earthMaterial);
    SH_scene.add(SH_earth);
    console.log("SH: Earth mesh added.");

    // Create ice layer (prefixed variable)
    // Use a slightly larger radius for the ice layer to avoid z-fighting
    const iceGeometry = new THREE.SphereGeometry(2.01, 64, 64); // Radius slightly larger than Earth
    // Ensure ice material allows transparency and doesn't fight with Earth rendering
    const iceMaterial = new THREE.MeshPhongMaterial({
        transparent: true,
        opacity: 0.9,
        depthWrite: false, // Important to see Earth through transparent parts
        blending: THREE.NormalBlending // Standard blending
    });
    SH_iceLayer = new THREE.Mesh(iceGeometry, iceMaterial);
    SH_scene.add(SH_iceLayer);
    console.log("SH: Ice layer mesh added.");

    // Create controls (prefixed variable)
    SH_controls = new OrbitControls(SH_camera, SH_renderer.domElement);
    // Target the South Pole area (adjust Y slightly if needed)
    SH_controls.target.set(0, -1, -0.03);
    SH_controls.enableDamping = true; // Optional: Smooths out controls
    SH_controls.dampingFactor = 0.1;
    SH_controls.rotateSpeed = 0.5;
    console.log("SH: OrbitControls created.");


    window.addEventListener('resize', SH_onWindowResize);
    console.log("SH: Resize listener added.");

    SH_animate();
    console.log("SH: Animation loop started via SH_animate().");

  } catch (error) {
      console.error("SH: Error during initScene:", error);
      container.innerHTML = `<p style="color: red; padding: 10px; font-family: sans-serif;">Error initializing visualization: ${error.message}. Check console (F12) for details.</p>`;
  }
}

// Handle window resize (prefixed function)
function SH_onWindowResize() {
  // Target the correct container ID
  const container = document.getElementById('sea-ice-SH-visualization');
  if (!container || !SH_camera || !SH_renderer) return; // Add checks for camera/renderer

  // Use container dimensions, fallback to 1 if zero
  const width = container.clientWidth || 1;
  const height = container.clientHeight || 1;
  console.log(`SH: Resizing to ${width}x${height}`); // Log resize

  SH_camera.aspect = width / height;
  SH_camera.updateProjectionMatrix();
  SH_renderer.setSize(width, height);
}

// Animation loop (prefixed function)
function SH_animate() {
  requestAnimationFrame(SH_animate);

  // Update controls each frame
  if (SH_controls) SH_controls.update();

  // FADING LOGIC START
  const FADE_RATE = 0.004; // How much opacity decreases each frame
  const REMOVE_THRESHOLD = 0.01; // Opacity level below which tubes are removed

  // Iterate through all years that have outlines
  Object.keys(SH_outlineTubes).forEach(year => {
    const tubes = SH_outlineTubes[year];
    const remainingTubes = []; // Keep track of tubes that are still visible

    if (Array.isArray(tubes)) {
      tubes.forEach(tube => {
        if (tube && tube.material) {
          // Decrease opacity
          tube.material.opacity -= FADE_RATE;

          // Check if tube should be kept or removed
          if (tube.material.opacity > REMOVE_THRESHOLD) {
            remainingTubes.push(tube); // Keep this tube
          } else {
            // Opacity is too low, remove the tube
            if (SH_scene) SH_scene.remove(tube);
            if (tube.geometry) tube.geometry.dispose();
            if (tube.material) tube.material.dispose();
            // console.log(`SH: Removed faded outline tube for year ${year}`); // Optional log
          }
        }
      });

      // Update the array for the year with only the remaining tubes
      SH_outlineTubes[year] = remainingTubes;

      // Optional: Clean up the year entry if no tubes remain
      if (remainingTubes.length === 0) {
        delete SH_outlineTubes[year];
        // console.log(`SH: Removed year ${year} from outline tracking.`); // Optional log
      }
    }
  });
  // FADING LOGIC END


  if (SH_renderer && SH_scene && SH_camera) {
    try {
        SH_renderer.render(SH_scene, SH_camera);
    } catch (renderError) {
        console.error("SH: Error during render:", renderError);
    }
  }
}

// Reset outlines (prefixed function)
function SH_resetOutlines() {
  // Reset outlines for all years
  SH_outlineTubes = {};
  // Reset hasResetOutlines flag
  SH_hasResetOutlines = true;
  console.log("SH: Outlines reset.");
}