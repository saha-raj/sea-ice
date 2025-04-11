/**
 * Sea Ice Visualization - Earth Globe Component
 * 
 * This file handles the creation and management of the Earth globe,
 * including loading textures and configuring surface properties.
 */

// Create the Earth globe with optional textures
function createEarthGlobe(scene, radius = 2) {
  // Create the Earth geometry
  const earthGeometry = new THREE.SphereGeometry(radius, 64, 64);
  
  // Create the Earth material
  // For simplicity in this initial implementation, we'll use a basic blue material
  // In a full implementation, we would add proper Earth textures
  const earthMaterial = new THREE.MeshPhongMaterial({
    color: 0x2233ff, // Ocean blue
    shininess: 25
  });
  
  // Create the Earth mesh
  const earthMesh = new THREE.Mesh(earthGeometry, earthMaterial);
  
  // Add the Earth to the scene
  scene.add(earthMesh);
  
  // Create a simple atmosphere effect
  createAtmosphere(scene, radius);
  
  return earthMesh;
}

// Create a simple atmosphere glow effect
function createAtmosphere(scene, radius) {
  // Create a slightly larger sphere for the atmosphere
  const atmosphereGeometry = new THREE.SphereGeometry(radius * 1.01, 64, 64);
  
  // Create the atmosphere material
  const atmosphereMaterial = new THREE.MeshBasicMaterial({
    color: 0x4464ff,
    transparent: true,
    opacity: 0.3,
    side: THREE.BackSide // Render on inside of sphere
  });
  
  // Create the atmosphere mesh
  const atmosphereMesh = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
  
  // Add the atmosphere to the scene
  scene.add(atmosphereMesh);
  
  return atmosphereMesh;
}

// Add a gentle rotation animation to the Earth
function animateEarthRotation(earthMesh, speed = 0.0002) {
  // Update function to be called in the animation loop
  return function update() {
    if (earthMesh) {
      earthMesh.rotation.y += speed;
    }
  };
}

// Update Earth globe position and rotation
function updateEarthPosition(earthMesh, position) {
  if (earthMesh && position) {
    earthMesh.position.copy(position);
  }
} 