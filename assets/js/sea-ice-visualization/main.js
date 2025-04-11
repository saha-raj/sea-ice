/**
 * Sea Ice 3D Visualization - Main Entry Point
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Global variables
let scene, camera, renderer;
let earth;
let controls;

// Initialize when the document is loaded
document.addEventListener('DOMContentLoaded', function() {
  initScene();
});

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
  camera.position.set(0, 3, 2);
  
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
  
  // Create ice layer
  const iceGeometry = new THREE.SphereGeometry(2, 64, 64);
  const iceTexture = new THREE.TextureLoader().load('output_textures/sea_ice_20200923_2048x1024_netcdf_global_nh_only.png');
  iceTexture.minFilter = THREE.NearestFilter;
  iceTexture.magFilter = THREE.NearestFilter;
  const iceMaterial = new THREE.MeshPhongMaterial({
    map: iceTexture,
    transparent: true,
    depthWrite: false,
    blending: THREE.NormalBlending
  });
  const iceMesh = new THREE.Mesh(iceGeometry, iceMaterial);
  scene.add(iceMesh);
  
  // Create controls
  controls = new OrbitControls(camera, renderer.domElement);
  
  // Start animation loop
  animate();
}

// Animation loop
function animate() {
  requestAnimationFrame(animate);
  if (controls) controls.update();
  if (renderer && scene && camera) {
    renderer.render(scene, camera);
  }
} 