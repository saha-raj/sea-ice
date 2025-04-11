import xarray as xr
import rioxarray  # Extends xarray with .rio accessor - IMPORTANT to import
import numpy as np
from PIL import Image
from pathlib import Path
from rasterio.enums import Resampling  # For specifying resampling method
from affine import Affine
import re  # Import regular expressions for filename parsing
from scipy import ndimage  # Import scipy for dilation, labeling, and center of mass
from skimage import measure, morphology  # For contour finding and smoothing
import json  # For saving GeoJSON
import glob  # For finding all .nc files
import argparse  # For command line arguments
import datetime  # For date validation and comparison

# --- Configuration ---
INPUT_DIR = Path("data/sea-ice-SH-bremen")
IMAGES_OUTPUT_DIR = Path("data/sea-ice-SH-images")
OUTLINES_OUTPUT_DIR = Path("data/sea-ice-SH-outlines")
RASTER_WIDTH = 2048
RASTER_HEIGHT = 1024
TARGET_CRS = "EPSG:4326"  # Target projection (lat/lon)

# Define colors for mapping (RGBA)
COLOR_ZERO_ICE = [0, 95, 153, 0]  # Blueish, transparent for zero ice concentration
COLOR_MAX_ICE = [255, 255, 255, 255]  # White, opaque for max ice concentration (1.0)

# --- Configuration for Bordering ---
HIGH_CONC_THRESHOLD = 0.7  # Ice above this threshold triggers border generation
# --- Alpha Gradient for Border Layers (Fraction: 1.0 = opaque, 0.0 = transparent) ---
# Generate 3 steps fading from 1.0 down to 0.2
BORDER_ALPHA_VALUES = np.linspace(1.0, 0.1, 3)
DILATION_ITERATIONS = len(BORDER_ALPHA_VALUES)  # Set iterations based on alpha values

# --- Calculate Fixed RGB Color for Border Pixels ---
# Interpolate color based on HIGH_CONC_THRESHOLD using the NEW color range
c_zero_rgb = np.array(COLOR_ZERO_ICE[:3])  # Use RGB from the new zero color
c_max_rgb = np.array(COLOR_MAX_ICE[:3])  # Use RGB from the max color
# Calculate the color corresponding to the threshold value
border_color_rgb_float = c_zero_rgb + HIGH_CONC_THRESHOLD * (c_max_rgb - c_zero_rgb)
# Ensure values are valid byte colors (0-255)
BORDER_COLOR_RGB = np.clip(border_color_rgb_float, 0, 255).astype(np.uint8)
print(f"Calculated fixed border color (RGB): {BORDER_COLOR_RGB}")

# --- Configuration for Pole Hole Filling ---
POLE_REGION_ROWS = RASTER_HEIGHT // 10  # Check top 10% of rows for the hole
MIN_CONC_FOR_ICE = 1e-6  # Threshold below which data is considered "non-ice" for hole detection

# Function to extract date from filename
def extract_date_from_filename(filename):
    # Looking for pattern like asi-AMSR2-n6250-20150115-v5.4.nc
    match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if match:
        year = match.group(1)
        month = match.group(2)
        day = match.group(3)
        return f"{year}-{month}-{day}", year, month
    return None, None, None

# --- Helper Function ---
def save_rgba_png(image_array, output_path):
    """Saves a NumPy RGBA array as a PNG image using Pillow."""
    try:
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        img = Image.fromarray(image_array, "RGBA")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        print(f"Successfully saved image to {output_path}")
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")

# --- Main Processing Function ---
def process_netcdf_file(input_file, output_image_path, output_outline_path):
    print(f"Processing {input_file}...")
    
    ds = None  # Initialize ds to None for finally block
    try:
        # --- 1. Open NetCDF and Select Data ---
        # mask_and_scale=True handles _FillValue (NaNf) automatically
        ds = xr.open_dataset(input_file, mask_and_scale=True, decode_coords="all")

        # Change variable name to 'z'
        data_var_name = "z"
        if data_var_name not in ds:
            raise ValueError(
                f"Variable '{data_var_name}' not found in the NetCDF file."
            )

        ice_conc_raw = ds[data_var_name]
        # Squeeze dimensions if necessary (though this file likely only has x, y)
        # Check for dimensions other than x and y and squeeze them if they are size 1
        dims_to_squeeze = [
            dim
            for dim in ice_conc_raw.dims
            if dim not in ["x", "y"] and ice_conc_raw.sizes[dim] == 1
        ]
        if dims_to_squeeze:
            ice_conc_raw = ice_conc_raw.squeeze(dims_to_squeeze, drop=True)

        # --- 2. Scale Data ---
        # Convert from percentage (0-100) to fraction (0-1)
        # NaNs will remain NaNs after division
        print("Scaling data from percentage (0-100) to fraction (0-1)...")
        ice_conc_scaled = ice_conc_raw / 100.0

        # --- 3. Set CRS and Reproject ---
        # Attempt to parse CRS automatically from grid_mapping attribute
        # rioxarray usually handles this well if grid_mapping var exists
        ice_conc_scaled = ice_conc_scaled.rio.set_spatial_dims(
            x_dim="x", y_dim="y", inplace=True
        )
        # Check if CRS needs to be written explicitly
        if ice_conc_scaled.rio.crs is None:
            print(
                "CRS not automatically detected, attempting to write EPSG:3411 based on metadata..."
            )
            # Try writing CRS using the grid_mapping variable name
            try:
                ice_conc_scaled = ice_conc_scaled.rio.write_crs(
                    ds["polar_stereographic"].attrs["spatial_ref"], inplace=True
                )
            except Exception:
                print(
                    "Failed to write CRS from 'spatial_ref', falling back to EPSG:3411."
                )
                ice_conc_scaled = ice_conc_scaled.rio.write_crs(
                    "EPSG:3411", inplace=True
                )

        # Ensure spatial dimensions are correctly named if needed (should be 'x', 'y' already)
        # This check might be redundant if set_spatial_dims worked, but safe to keep
        if "x" not in ice_conc_scaled.dims or "y" not in ice_conc_scaled.dims:
            raise ValueError("Spatial dimensions 'x' and 'y' not found after setting.")

        print(f"Source CRS: {ice_conc_scaled.rio.crs}")
        print(
            f"Reprojecting to {TARGET_CRS} ({RASTER_WIDTH}x{RASTER_HEIGHT}) covering full globe..."
        )

        # --- Define Full Global Extent for Target ---
        lon_res = 360.0 / RASTER_WIDTH
        lat_res = 180.0 / RASTER_HEIGHT
        target_transform = Affine(lon_res, 0.0, -180.0, 0.0, -lat_res, 90.0)

        # Reproject to the target CRS, specifying full global transform and shape
        # Use the scaled data (ice_conc_scaled)
        ice_conc_reprojected = ice_conc_scaled.rio.reproject(
            dst_crs=TARGET_CRS,
            shape=(RASTER_HEIGHT, RASTER_WIDTH),
            transform=target_transform,
            resampling=Resampling.bilinear,
            nodata=np.nan,  # NaNs from _FillValue or scaling will be handled
        )
        print("Reprojection complete.")

        # Extract the data as a NumPy array (shape will be H, W)
        final_grid = ice_conc_reprojected.values

        # --- NEW Step 3a: Fill Pole Hole Artifact ---
        print("Attempting to fill pole hole artifact...")
        grid_with_hole_filled = final_grid.copy()  # Start with a copy
        pole_hole_filled = False

        try:
            # Create mask of non-ice pixels (NaN or near-zero) in the pole region
            is_non_ice = np.isnan(final_grid[:POLE_REGION_ROWS, :]) | (final_grid[:POLE_REGION_ROWS, :] <= MIN_CONC_FOR_ICE)

            # Label connected components of non-ice in the pole region
            # Use a simple 4-connectivity structure
            structure = ndimage.generate_binary_structure(2, 1)
            labeled_array, num_features = ndimage.label(is_non_ice, structure=structure)

            if num_features > 0:
                print(f"  Found {num_features} non-ice component(s) in the pole region.")
                # Calculate center of mass for each component
                centers = ndimage.center_of_mass(is_non_ice, labeled_array, range(1, num_features + 1))
                # Target center: top row, middle column
                target_center = (0, RASTER_WIDTH // 2)

                # Find the component closest to the target center
                closest_feature_label = -1
                min_dist_sq = float('inf')

                # Ensure centers is iterable (list of tuples) even if only one feature
                if num_features == 1:
                    centers = [centers]  # Make it a list containing the single tuple

                for i, center in enumerate(centers):
                    label = i + 1
                    dist_sq = (center[0] - target_center[0])**2 + (center[1] - target_center[1])**2
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        closest_feature_label = label

                if closest_feature_label != -1:
                    print(f"  Identified component {closest_feature_label} as potential pole hole.")
                    # Create mask for the identified pole hole component (only in the top rows)
                    pole_hole_mask_region = (labeled_array == closest_feature_label)
                    # Create a full-sized mask for dilation
                    pole_hole_mask_full = np.zeros_like(final_grid, dtype=bool)
                    pole_hole_mask_full[:POLE_REGION_ROWS, :] = pole_hole_mask_region

                    # Dilate the hole mask by 1 pixel to find the boundary
                    dilated_hole_mask = ndimage.binary_dilation(pole_hole_mask_full, structure=structure, iterations=1)
                    # Boundary pixels are in the dilated mask but not the original hole mask
                    hole_boundary_mask = dilated_hole_mask & ~pole_hole_mask_full

                    # Get concentration values from the original grid at the boundary
                    boundary_values = final_grid[hole_boundary_mask]
                    # Filter out any NaNs that might be on the boundary (unlikely but safe)
                    valid_boundary_values = boundary_values[~np.isnan(boundary_values)]

                    if valid_boundary_values.size > 0:
                        # Calculate the mean concentration of the boundary
                        hole_fill_value = np.mean(valid_boundary_values)
                        print(f"  Calculated fill value from boundary: {hole_fill_value:.4f}")
                        # Fill the hole in the copied grid
                        grid_with_hole_filled[pole_hole_mask_full] = hole_fill_value
                        pole_hole_filled = True
                        print("  Successfully filled pole hole.")
                    else:
                        print("  Warning: Could not find valid boundary values to calculate fill value.")
                else:
                    print("  Could not identify a closest component (this shouldn't happen if num_features > 0).")
            else:
                print("  No non-ice components found in the pole region. No hole to fill.")

        except Exception as e:
            print(f"  Error during pole hole filling: {e}. Proceeding without filling.")
            # Ensure grid_with_hole_filled is still the original grid copy
            grid_with_hole_filled = final_grid.copy()

        # --- Step 3b: Identify Border Layers (using the potentially filled grid) ---
        print(f"Identifying border layers for alpha gradient (using {'filled' if pole_hole_filled else 'original'} grid)...")

        # Use grid_with_hole_filled from now on
        # Mask of HIGH concentration ice pixels
        high_conc_mask = ~np.isnan(grid_with_hole_filled) & (grid_with_hole_filled > HIGH_CONC_THRESHOLD)
        # Mask of NaN pixels (in the potentially filled grid - should be fewer NaNs if hole was filled)
        # Note: We border based on NaN, so the filled hole won't get a border itself.
        nan_mask = np.isnan(grid_with_hole_filled)

        # Define the structuring element for 4-connectivity
        structure = ndimage.generate_binary_structure(2, 1)

        # Store the mask for each border layer
        border_layer_masks = []
        # Keep track of the combined mask of all filled border pixels found so far
        all_filled_border_mask = np.zeros_like(high_conc_mask, dtype=bool)
        # Keep track of the mask from the previous iteration's dilation result
        previous_iteration_mask = high_conc_mask.copy()

        # Iterate to find masks for each layer
        for i in range(DILATION_ITERATIONS):
            print(f"  Finding border layer {i+1}/{DILATION_ITERATIONS}...")

            # Dilate the mask from the *previous* iteration by one step
            current_dilated_mask = ndimage.binary_dilation(
                previous_iteration_mask,
                structure=structure,
                iterations=1  # Dilate only one step from previous
            )

            # Identify pixels added in *this specific* dilation iteration
            newly_added_mask = current_dilated_mask & ~previous_iteration_mask

            # Identify pixels in this new layer that were originally NaN (in grid_with_hole_filled)
            # AND have not already been assigned to an inner layer
            current_border_layer_mask = newly_added_mask & nan_mask & ~all_filled_border_mask

            border_layer_masks.append(current_border_layer_mask)
            print(f"    Found {np.sum(current_border_layer_mask)} pixels for layer {i+1}.")

            # Update the mask tracking all filled pixels
            all_filled_border_mask |= current_border_layer_mask
            # Update the mask for the next iteration
            previous_iteration_mask = current_dilated_mask

        print(f"Finished identifying border layers. Total border pixels: {np.sum(all_filled_border_mask)}")

        # --- 4. Create RGBA Image Array ---
        print("Creating RGBA image array...")
        rgba_image = np.zeros((RASTER_HEIGHT, RASTER_WIDTH, 4), dtype=np.uint8)
        # Initialize fully transparent (alpha=0 based on COLOR_ZERO_ICE)
        rgba_image[:, :, 3] = COLOR_ZERO_ICE[3]

        # --- 5. Apply Color and Alpha Mapping ---
        print("Applying border alpha gradient and original ice colors...")

        # Apply border layers first, from outermost to innermost
        # Use the pre-calculated BORDER_COLOR_RGB and gradient alpha
        for i in range(DILATION_ITERATIONS - 1, -1, -1):
            layer_mask = border_layer_masks[i]
            alpha_fraction = BORDER_ALPHA_VALUES[i]
            # Convert alpha fraction (0.0-1.0) to byte (0-255)
            alpha_byte = np.clip(int(round(alpha_fraction * 255)), 0, 255)
            print(f"  Applying Border Layer {i+1}: Alpha={alpha_byte}, Color={BORDER_COLOR_RGB}")

            # Set the fixed RGB color for this layer's pixels
            rgba_image[layer_mask, 0:3] = BORDER_COLOR_RGB
            # Set the calculated Alpha for this layer's pixels
            rgba_image[layer_mask, 3] = alpha_byte

        # Apply original ice colors and full opacity *last*
        # Use the grid_with_hole_filled for concentration values
        print(f"  Applying ice data from {'filled' if pole_hole_filled else 'original'} grid (interpolated color, fully opaque)...")
        # Find pixels with valid concentration in the *potentially filled* grid
        valid_mask_original = ~np.isnan(grid_with_hole_filled) & (grid_with_hole_filled > MIN_CONC_FOR_ICE)

        # Get concentration values for valid pixels from the *potentially filled* grid (clamp 0-1)
        concentration = np.clip(grid_with_hole_filled[valid_mask_original], 0.0, 1.0)

        # Calculate RGB based on concentration gradient between NEW COLOR_ZERO_ICE and COLOR_MAX_ICE
        # Using c_zero_rgb and c_max_rgb defined in config section
        interpolated_rgb_float = c_zero_rgb + concentration[:, np.newaxis] * (c_max_rgb - c_zero_rgb)
        interpolated_rgb_byte = np.clip(interpolated_rgb_float, 0, 255).astype(np.uint8)

        # Apply interpolated RGB color to the valid original ice pixels
        rgba_image[valid_mask_original, 0:3] = interpolated_rgb_byte
        # Set Alpha to fully opaque using the alpha from COLOR_MAX_ICE (which is 255)
        rgba_image[valid_mask_original, 3] = COLOR_MAX_ICE[3]

        print("RGBA image array created.")

        # --- 6. Save Image ---
        print(f"Saving image to: {output_image_path}")
        save_rgba_png(rgba_image, output_image_path)

        # --- 7. Extract and Save Ice Outline ---
        print("Extracting smooth outline of largest ice mass...")
        
        # Create binary mask of high concentration ice
        ice_mask = ~np.isnan(grid_with_hole_filled) & (grid_with_hole_filled > HIGH_CONC_THRESHOLD)
        
        # Label connected components and find the largest
        labels, num_features = ndimage.label(ice_mask)
        if num_features > 0:
            # Find sizes of all features
            sizes = ndimage.sum(ice_mask, labels, range(1, num_features + 1))
            largest_feature = np.argmax(sizes) + 1
            
            # Keep only the largest feature
            largest_mask = (labels == largest_feature)
            
            # Apply some morphological operations to smooth the mask
            largest_mask = morphology.binary_closing(largest_mask)
            largest_mask = morphology.binary_opening(largest_mask)
            
            # Find contours of the smoothed mask
            contours = measure.find_contours(largest_mask.astype(float), 0.5)
            
            if contours:
                # Get the longest contour (main outline)
                longest_contour = max(contours, key=len)
                
                # Convert pixel coordinates to lon/lat
                lon_coords = -180 + (longest_contour[:, 1] / RASTER_WIDTH) * 360
                lat_coords = 90 - (longest_contour[:, 0] / RASTER_HEIGHT) * 180
                
                # Create coordinate pairs for GeoJSON
                coordinates = np.column_stack([lon_coords, lat_coords])
                
                # Reduce number of points while preserving shape
                # Take every Nth point (adjust N to balance detail and file size)
                N = 5
                coordinates = coordinates[::N]
                
                # Create GeoJSON feature
                geojson = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coordinates.tolist()
                    },
                    "properties": {
                        "date": Path(input_file).stem.split('-')[3],  # Extract date from filename
                        "threshold": HIGH_CONC_THRESHOLD
                    }
                }
                
                # Save outline
                print(f"Saving outline to: {output_outline_path}")
                with open(output_outline_path, 'w') as f:
                    json.dump(geojson, f)
                print("Outline saved successfully.")
            else:
                print("No contours found in the largest ice mass.")
        else:
            print("No ice masses found above threshold.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()  # Uncomment for detailed traceback during debugging
    finally:
        # Close the dataset if it was opened
        if ds is not None:
            ds.close()

# --- Main Script ---
if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Batch process NetCDF sea ice files to PNG and GeoJSON')
    parser.add_argument('--start-year', type=int, help='Start year (YYYY)')
    parser.add_argument('--start-month', type=int, help='Start month (1-12)')
    parser.add_argument('--end-year', type=int, help='End year (YYYY)')
    parser.add_argument('--end-month', type=int, help='End month (1-12)')
    parser.add_argument('--day', type=int, default=15, help='Day of month to use (default: 15)')
    args = parser.parse_args()
    
    # Validate date range if provided
    date_filtering = False
    if any([args.start_year, args.start_month, args.end_year, args.end_month]):
        # If any date parameter is provided, all start and end parameters must be provided
        if not all([args.start_year, args.start_month, args.end_year, args.end_month]):
            parser.error("If filtering by date, all parameters (start-year, start-month, end-year, end-month) must be provided")
        
        # Validate month ranges
        if not (1 <= args.start_month <= 12 and 1 <= args.end_month <= 12):
            parser.error("Months must be between 1 and 12")
            
        # Create date objects for comparison
        try:
            start_date = datetime.date(args.start_year, args.start_month, args.day)
            end_date = datetime.date(args.end_year, args.end_month, args.day)
            
            # Check that end date is after start date
            if end_date < start_date:
                parser.error("End date must be after start date")
                
            date_filtering = True
            print(f"Filtering files from {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
        except ValueError as e:
            parser.error(f"Invalid date: {e}")
    else:
        print("No date range specified, processing all files")
    
    print("Starting batch processing of NetCDF files...")
    
    # Create output directories
    IMAGES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTLINES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all NetCDF files in the input directory
    netcdf_files = list(INPUT_DIR.glob("*.nc"))
    print(f"Found {len(netcdf_files)} NetCDF files to process")
    
    # Filter files by date if needed
    filtered_files = []
    if date_filtering:
        for input_file in netcdf_files:
            date_str, year_str, month_str = extract_date_from_filename(input_file.name)
            if not date_str:
                continue
                
            try:
                # Extract year and month as integers
                year = int(year_str)
                month = int(month_str)
                
                # Check if this file is within our date range
                file_date = datetime.date(year, month, args.day)
                if start_date <= file_date <= end_date:
                    filtered_files.append(input_file)
            except (ValueError, TypeError):
                # Skip files with invalid dates
                continue
        
        print(f"Filtered to {len(filtered_files)} files within date range")
        netcdf_files = filtered_files
    
    # Process each file
    for i, input_file in enumerate(netcdf_files):
        print(f"\nProcessing file {i+1}/{len(netcdf_files)}: {input_file}")
        
        # Extract date from filename
        date_str, _, _ = extract_date_from_filename(input_file.name)
        if not date_str:
            print(f"Could not extract date from filename: {input_file.name}, skipping")
            continue
        
        # Set output paths
        image_output_path = IMAGES_OUTPUT_DIR / f"{date_str}.png"
        outline_output_path = OUTLINES_OUTPUT_DIR / f"{date_str}.geojson"
        
        # Process the file
        process_netcdf_file(input_file, image_output_path, outline_output_path)
    
    print("\nBatch processing complete!") 