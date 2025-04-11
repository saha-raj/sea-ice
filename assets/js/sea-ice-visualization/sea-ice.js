/**
 * Sea Ice Visualization - GeoJSON to 3D Conversion
 * 
 * This file handles the conversion of GeoJSON sea ice data
 * to 3D mesh objects that can be rendered on the Earth globe.
 */

// Convert latitude/longitude to 3D position on a sphere
function latLongToVector3(lat, lon, radius) {
  // Convert to radians
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  
  // Calculate 3D position
  const x = -radius * Math.sin(phi) * Math.cos(theta);
  const y = radius * Math.cos(phi);
  const z = radius * Math.sin(phi) * Math.sin(theta);
  
  return new THREE.Vector3(x, y, z);
}

// Create mesh from GeoJSON sea ice data
function createIceMeshFromGeoJSON(geoJson, radius = 2.02) {
  // Create a container for all ice meshes
  const iceObject = new THREE.Object3D();
  
  // Material for ice
  const iceMaterial = new THREE.MeshPhongMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.8,
    side: THREE.DoubleSide,
    shininess: 50
  });
  
  // Process each feature in the GeoJSON
  if (geoJson && geoJson.features) {
    geoJson.features.forEach(feature => {
      if (feature.geometry && feature.geometry.coordinates) {
        const coordinates = feature.geometry.coordinates;
        
        // Process each polygon
        coordinates.forEach(polygon => {
          // Each polygon may have multiple rings (outer + holes)
          polygon.forEach(ring => {
            const vertices = [];
            const triangles = [];
            
            // Convert lat/lon points to 3D positions
            ring.forEach((coord, index) => {
              const lon = coord[0];
              const lat = coord[1];
              const pos = latLongToVector3(lat, lon, radius);
              vertices.push(pos.x, pos.y, pos.z);
            });
            
            // Create triangles (simple triangulation for convex polygons)
            for (let i = 1; i < ring.length - 1; i++) {
              triangles.push(0, i, i + 1);
            }
            
            // Create buffer geometry
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
            geometry.setIndex(triangles);
            geometry.computeVertexNormals();
            
            // Create mesh and add to container
            const mesh = new THREE.Mesh(geometry, iceMaterial);
            iceObject.add(mesh);
          });
        });
      }
    });
  }
  
  return iceObject;
}

// Process a month of sea ice data
function processMonthlyIceData(geoJson, radius = 2.02) {
  const iceObject = createIceMeshFromGeoJSON(geoJson, radius);
  return iceObject;
}

// Animate between two ice states
function animateBetweenIceStates(startState, endState, duration = 1000) {
  // This is a placeholder for animation logic
  // In a full implementation, we would:
  // 1. Interpolate between the two states
  // 2. Update the mesh at each animation frame
  
  return {
    // Return animation control functions
    start: function() {
      // Start the animation
      console.log('Animation started');
    },
    stop: function() {
      // Stop the animation
      console.log('Animation stopped');
    }
  };
} 