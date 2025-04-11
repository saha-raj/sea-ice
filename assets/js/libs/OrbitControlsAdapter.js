/**
 * OrbitControls Adapter
 * 
 * This file adapts the ES6 module version of OrbitControls to work with
 * the global THREE object pattern. This is needed because the downloaded
 * OrbitControls.js uses ES6 imports, but we're using the global THREE object.
 */

// Wait for THREE to be available
if (typeof THREE === 'undefined') {
  console.error('THREE is not defined. Make sure three.min.js is loaded before this script.');
}

// Create a global variable to store the OrbitControls class
THREE.OrbitControls = (function() {
  // Define the classes and constants needed by OrbitControls
  
  // Mouse constants
  if (!THREE.MOUSE) {
    THREE.MOUSE = {
      LEFT: 0,
      MIDDLE: 1,
      RIGHT: 2
    };
  }
  
  // Touch constants
  if (!THREE.TOUCH) {
    THREE.TOUCH = {
      ROTATE: 0,
      DOLLY_PAN: 1
    };
  }
  
  // OrbitControls implementation
  class OrbitControls {
    constructor(object, domElement) {
      this.object = object;
      this.domElement = domElement || document;
      
      // Set defaults
      this.enabled = true;
      this.target = new THREE.Vector3();
      
      this.minDistance = 0;
      this.maxDistance = Infinity;
      
      this.minZoom = 0;
      this.maxZoom = Infinity;
      
      this.minPolarAngle = 0; // radians
      this.maxPolarAngle = Math.PI; // radians
      
      this.minAzimuthAngle = -Infinity; // radians
      this.maxAzimuthAngle = Infinity; // radians
      
      this.enableDamping = false;
      this.dampingFactor = 0.05;
      
      this.enableZoom = true;
      this.zoomSpeed = 1.0;
      
      this.enableRotate = true;
      this.rotateSpeed = 1.0;
      
      this.enablePan = true;
      this.panSpeed = 1.0;
      this.screenSpacePanning = true;
      this.keyPanSpeed = 7.0;
      
      this.autoRotate = false;
      this.autoRotateSpeed = 2.0;
      
      this.keys = {
        LEFT: 'ArrowLeft',
        UP: 'ArrowUp',
        RIGHT: 'ArrowRight',
        BOTTOM: 'ArrowDown'
      };
      
      this.mouseButtons = {
        LEFT: THREE.MOUSE.ROTATE,
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: THREE.MOUSE.PAN
      };
      
      // Internal state
      this._state = 0; // none
      this._prevState = 0;
      
      this._spherical = new THREE.Spherical();
      this._sphericalDelta = new THREE.Spherical();
      
      this._scale = 1;
      this._panOffset = new THREE.Vector3();
      this._zoomChanged = false;
      
      this._rotateStart = new THREE.Vector2();
      this._rotateEnd = new THREE.Vector2();
      this._rotateDelta = new THREE.Vector2();
      
      this._panStart = new THREE.Vector2();
      this._panEnd = new THREE.Vector2();
      this._panDelta = new THREE.Vector2();
      
      this._dollyStart = new THREE.Vector2();
      this._dollyEnd = new THREE.Vector2();
      this._dollyDelta = new THREE.Vector2();
      
      // Event handlers and setup
      this._onContextMenu = this._onContextMenu.bind(this);
      this._onMouseDown = this._onMouseDown.bind(this);
      this._onMouseMove = this._onMouseMove.bind(this);
      this._onMouseUp = this._onMouseUp.bind(this);
      this._onMouseWheel = this._onMouseWheel.bind(this);
      this._onKeyDown = this._onKeyDown.bind(this);
      
      this.domElement.addEventListener('contextmenu', this._onContextMenu);
      this.domElement.addEventListener('pointerdown', this._onMouseDown);
      this.domElement.addEventListener('wheel', this._onMouseWheel, { passive: false });
      
      // Initialize
      this.update();
    }
    
    // Simplified update method (full implementation would be much larger)
    update() {
      // Update the camera position based on internal state
      const offset = new THREE.Vector3();
      const quat = new THREE.Quaternion().setFromUnitVectors(
        this.object.up, new THREE.Vector3(0, 1, 0)
      );
      const quatInverse = quat.clone().invert();
      
      offset.copy(this.object.position).sub(this.target);
      offset.applyQuaternion(quat);
      
      // Apply rotation and scale
      if (this.autoRotate) {
        this._rotateLeft(this._getAutoRotationAngle());
      }
      
      offset.x += this._panOffset.x;
      offset.y += this._panOffset.y;
      offset.z += this._panOffset.z;
      
      offset.applyQuaternion(quatInverse);
      
      this.object.position.copy(this.target).add(offset);
      this.object.lookAt(this.target);
      
      // Reset state
      this._sphericalDelta.set(0, 0, 0);
      this._panOffset.set(0, 0, 0);
      this._scale = 1;
      
      return true;
    }
    
    // Placeholder methods (actual implementation would be much more complex)
    _getAutoRotationAngle() {
      return (2 * Math.PI / 60 / 60) * this.autoRotateSpeed;
    }
    
    _rotateLeft(angle) {
      this._sphericalDelta.theta -= angle;
    }
    
    _onContextMenu(event) {
      if (!this.enabled) return;
      event.preventDefault();
    }
    
    _onMouseDown(event) {
      if (!this.enabled) return;
      event.preventDefault();
      
      // Set states and store start positions
      // Full implementation would handle multiple buttons and modes
    }
    
    _onMouseMove(event) {
      if (!this.enabled) return;
      event.preventDefault();
      
      // Handle rotation, panning, etc. based on the current state
    }
    
    _onMouseUp(event) {
      // Reset states
    }
    
    _onMouseWheel(event) {
      if (!this.enabled || !this.enableZoom) return;
      event.preventDefault();
      
      // Handle zooming
    }
    
    _onKeyDown(event) {
      if (!this.enabled || !this.enablePan) return;
      
      // Handle keyboard panning
    }
    
    dispose() {
      this.domElement.removeEventListener('contextmenu', this._onContextMenu);
      this.domElement.removeEventListener('pointerdown', this._onMouseDown);
      this.domElement.removeEventListener('wheel', this._onMouseWheel);
      document.removeEventListener('pointermove', this._onMouseMove);
      document.removeEventListener('pointerup', this._onMouseUp);
    }
  }
  
  return OrbitControls;
})(); 