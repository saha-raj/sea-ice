#!/usr/bin/env python3

import json
import numpy as np
from PIL import Image
import os
from pathlib import Path
import glob
from scipy.ndimage import binary_dilation, gaussian_filter

def create_raster_from_points(features, width=2048, height=1024):
    """
    Create a continuous ice surface by plotting points and then dilating/smoothing.
    Width 2048 and height 1024 gives us high resolution output.
    Grid starts at 180°W (left edge) and goes to 180°E (right edge).
    """
    # Create empty grid
    grid = np.zeros((height, width), dtype=np.float32)
    
    # Calculate conversion factors
    lon_to_x = width / 360.0    # pixels per degree longitude
    lat_to_y = height / 180.0   # pixels per degree latitude
    
    # Plot points
    for feature in features:
        if feature['geometry']['type'] == 'Point':
            lon, lat = feature['geometry']['coordinates']
            # Concentration is already 0-1, just use it directly
            concentration = float(feature['properties']['concentration'])
            
            # Convert lat/lon to grid coordinates
            x = int(((lon + 180) * lon_to_x + 0.5) % width)
            y = int(((90 - lat) * lat_to_y) + 0.5)
            
            if 0 <= x < width and 0 <= y < height:
                grid[y, x] = concentration

    # Create binary mask of where we have data (any non-zero concentration)
    mask = grid > 0.01  # Use small threshold to avoid numerical issues
    
    # Dilate the mask to fill gaps between points
    dilated_mask = binary_dilation(mask, iterations=3)
    
    # Smooth the concentration values
    smoothed = gaussian_filter(grid, sigma=3)
    
    # Use the dilated mask to keep only the ice-covered regions
    final = np.where(dilated_mask, smoothed, 0)
    
    # Ensure values stay in 0-1 range
    final = np.clip(final, 0, 1)
    
    return final

def save_raster(grid, output_path):
    """
    Save the numpy grid as a grayscale PNG.
    Values are scaled to 0-255 range.
    """
    # Scale 0-1 to 0-255
    scaled = (grid * 255).astype(np.uint8)
    
    # Create and save image
    img = Image.fromarray(scaled, mode='L')
    img.save(output_path)

def process_all_files():
    # Create output directory if it doesn't exist
    output_dir = Path('data/seaice_geojson/rasters')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all JSON files
    json_files = glob.glob('data/seaice_geojson/_all/*.json')
    
    for json_file in json_files:
        print(f"Processing {json_file}...")
        
        # Skip the yearly summary files
        if json_file.endswith(('2002.json', '2003.json')):
            continue
        
        # Get output filename
        base_name = os.path.basename(json_file)
        output_name = base_name.replace('.json', '.png')
        output_path = output_dir / output_name
        
        # Read and process JSON
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Create raster
            grid = create_raster_from_points(data['features'])
            
            # Save as PNG
            save_raster(grid, output_path)
            print(f"Saved {output_path}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

if __name__ == '__main__':
    process_all_files() 